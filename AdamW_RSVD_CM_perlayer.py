import os
import argparse
import logging
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from datasets import DatasetDict, load_dataset
import transformers
from transformers import default_data_collator, get_cosine_schedule_with_warmup
from huggingface_hub import login, notebook_login
from lion_pytorch import Lion
from tqdm import tqdm

from Mylog import TitledLog
import Preprocessing
from Preprocessing import load_meta_math, MetaMathQA100k_Preprocessor
from new_optim import RSVD_CM_AdamW

import wandb

import math

log = logging.getLogger(__name__)

# disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_SILENT"] = "true"

torch.set_float32_matmul_precision("medium")

config = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 1,
    "learning_rate": 3e-5,
    "weight_decay": 0,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "bf16": True,
    "logging_steps": 1,
    "eval_steps": -1,  # 每个 epoch 结束后评估
}

def main():
  local_rank = int(os.getenv("LOCAL_RANK", "0"))
  world_size = int(os.getenv("WORLD_SIZE", "1"))
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if local_rank == 0:
        wandb.init(
            project='LLAMA-2-7B',
            name=f"llama-2-7b_math_Adam_RSVD_CM",
            group='llama-2-7B-Math',
        )

  model_name = "meta-llama/Llama-2-7b-chat-hf"
  tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
  if tokenizer.eos_token is None:
      tokenizer.add_special_tokens({"eos_token": "</s>"})
      model.resize_token_embeddings(len(tokenizer))
  if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
  
  model = transformers.LlamaForCausalLM.from_pretrained(model_name, max_length=1024,attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}) 
  model.config.use_cache = False
  model.gradient_checkpointing_enable()

  with TitledLog("load datasets and dataloaders", log_fn=log.info):
        datasets = load_meta_math()

        preprocessor = MetaMathQA100k_Preprocessor(
            tokenizer=tokenizer,
            tokenizer_kwargs={
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "pt",
                "max_length": 512
            },
        )

        datasets = datasets.map(
            preprocessor,
            batched=True,
            batch_size=1000,
            num_proc=1,
            desc="Running tokenizer on dataset",
        )

  train_loader = DataLoader(
    datasets["train"],
    batch_size=config["per_device_train_batch_size"],
    collate_fn=default_data_collator,
    shuffle=True
  )

  eval_loader = DataLoader(
    datasets["eval"],
    batch_size=config["per_device_eval_batch_size"],
    collate_fn=default_data_collator
  )

  # 初始化自定义优化器（保持与 CustomTrainer 一致）
  optimizer = RSVD_CM_AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
    rank=4
  )
  total_steps = len(train_loader) * config["num_train_epochs"]
  warmup_steps = int(total_steps * config["warmup_ratio"])

  # 创建学习率调度器（复现原始调度逻辑）
  scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
  )
  # 混合精度训练配置
  scaler = torch.cuda.amp.GradScaler(enabled=config["bf16"])

  # 训练循环
  model.train()
  global_step = 0

  for epoch in range(config["num_train_epochs"]):
      # 训练阶段
      progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
      for batch in progress_bar:
          # 将数据移至设备
          batch = {k: v.to(device) for k, v in batch.items()}
        
          # 混合精度前向传播
          with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config["bf16"]):
              outputs = model(**batch)
              loss = outputs.loss

          # 反向传播
          scaler.scale(loss).backward()
        
          # 参数更新
          scaler.step(optimizer)
          scaler.update()
          optimizer.zero_grad()
          scheduler.step()
        
          # 日志记录
          if global_step % config["logging_steps"] == 0:
              log_data = {
                  "loss": loss.item(),
                  "lr": scheduler.get_last_lr()[0],
                  "epoch": epoch + (global_step + 1) / len(train_loader)
              }
            
              if local_rank == 0:
                  wandb.log(log_data)
            
              progress_bar.set_postfix(loss=loss.item(), lr=log_data["lr"])
        
          global_step += 1

      # 评估阶段（每个 epoch 结束后）
      model.eval()
      eval_loss = 0
    
      with torch.no_grad():
          for batch in tqdm(eval_loader, desc="Evaluating"):
              batch = {k: v.to(device) for k, v in batch.items()}
            
              with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config["bf16"]):
                  outputs = model(**batch)
                  eval_loss += outputs.loss.item()
    
      eval_loss /= len(eval_loader)
    
      # 记录评估结果
      if local_rank == 0:
          wandb.log({"eval_loss": eval_loss, "epoch": epoch + 1})
          print(f"Epoch {epoch+1} Evaluation Loss: {eval_loss:.4f}")
  if local_rank == 0:
    model.save_pretrained(f'./logs/transformers/llama-2-7b/math/Adam_RSVD_CM_perlayer/lr_3e-5')
    tokenizer.save_pretrained(f'./logs/transformers/llama-2-7b/math/Adam_RSVD_CM_perlayer/lr_3e-5')

    wandb.finish()

if __name__ == "__main__":
    main()
