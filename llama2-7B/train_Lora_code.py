import os
import logging
import wandb
import math

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.optim as optim

from datasets import DatasetDict, load_dataset
import transformers
from transformers import default_data_collator, get_linear_schedule_with_warmup
from transformers import TrainingArguments, Trainer

from huggingface_hub import login, notebook_login
from tqdm import tqdm

import peft
from peft import LoraConfig, LoraRuntimeConfig
from peft.optimizers import create_loraplus_optimizer

from Mylog import TitledLog
import Preprocessing
from Preprocessing import load_codefeedback, CodeFeedback100k_Preprocessor




log = logging.getLogger(__name__)

# disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_SILENT"] = "true"

torch.set_float32_matmul_precision("medium")

config = {
    "num_train_epochs": 1,
    "per_device_train_batch_size":32,
    "rank":4,
    "per_device_eval_batch_size": 1,
    "learning_rate": 1e-3,
    "method": "default", # "default", "pissa" or "dora"
    "optimizer": "default", # "default" or "loraplus"
    "loraplus_lr_ratio": 4,
    "weight_decay": 0,
    "warmup_ratio": 0.03,
    "bf16": True,
    "logging_steps": 1,
    "eval_steps": -1,  # 每个 epoch 结束后评估
}

def main():
  local_rank = int(os.getenv("LOCAL_RANK", "0"))
  world_size = int(os.getenv("WORLD_SIZE", "1"))
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.cuda.manual_seed_all(0)
    
  if local_rank == 0:
        wandb.init(
            project='LLAMA-2-7B',
            name=f"llama-2-7b_code_lora_{config['method']}_{config['optimizer']}",
            group='llama-2-7B-Code',
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
  if config["method"] == "default":
      lora_config = LoraConfig(
        r=config["rank"],
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        )
  elif config["method"] == "pissa":
      lora_config = LoraConfig(
        init_lora_weights="pissa_niter_4",
        r=config["rank"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        )
  elif config["method"] == "dora":
      lora_config = LoraConfig(
        use_dora=True, 
        runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=True),
        r=config["rank"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        )
  else:
      raise RuntimeError("Incorrect lora method config")
      
  model = peft.get_peft_model(model, lora_config)
    
  with TitledLog("load datasets and dataloaders", log_fn=log.info):
      datasets = load_codefeedback()
        
      preprocessor = CodeFeedback100k_Preprocessor(
            tokenizer=tokenizer,
            tokenizer_kwargs={
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "pt",
                "max_length": 1024
            },
        )
      
      datasets = datasets.map(
            preprocessor,
            batched=True,
            batch_size=1000,
            num_proc=1,
            desc="Running tokenizer on dataset",
        )


  train_args = TrainingArguments(
        output_dir="./llama-2-7b-codefeedback100k",
        run_name = "llama2_7b_code_experiment",
        logging_dir="./llama-2-7b-codefeedback100k_logs",
        do_train=True,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        logging_steps=config["logging_steps"],
        bf16=config["bf16"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type="linear",
        report_to="wandb" if local_rank == 0 else None,
        label_names=[
            "labels"
        ],
        do_eval=True,
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        ddp_find_unused_parameters=False,
        evaluation_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="no",
    )

  class CustomTrainer(Trainer):
    def create_optimizer(self):
        if config["optimizer"] == "default":
            self.optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        elif config["optimizer"] == "loraplus":
            self.optimizer = create_loraplus_optimizer(model=model, optimizer_cls= optim.AdamW, lr=config["learning_rate"], loraplus_lr_ratio=config["loraplus_lr_ratio"], weight_decay=config["weight_decay"])
        else:
            raise RuntimeError("Incorrect optimizer config")
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        仍然使用 Trainer 默认的 lr_scheduler
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = super().create_scheduler(num_training_steps, self.optimizer)
        return self.lr_scheduler
        
  trainer = CustomTrainer(
      model=model,
      args=train_args,
      train_dataset=datasets["train"],
      eval_dataset=datasets["eval"],
      data_collator=default_data_collator,
  )
  trainer.train()
    
  if local_rank == 0:
      if config["method"] == "pissa":
          model.peft_config["default"].init_lora_weights = True
          model.save_pretrained(f'./logs/transformers/llama-2-7b/code/Lora_adapter/method_{config["method"]}/optimizer_{config["optimizer"]}/lr_{config["learning_rate"]}')
          model = model.unload()
          model.save_pretrained(f'./logs/transformers/llama-2-7b/code/Lora_adapter/method_{config["method"]}/optimizer_{config["optimizer"]}/lr_{config["learning_rate"]}/pissa_residual_model')
          tokenizer.save_pretrained(f'./logs/transformers/llama-2-7b/code/Lora_adapter/method_{config["method"]}/optimizer_{config["optimizer"]}/lr_{config["learning_rate"]}')
      else:  
          model.save_pretrained(f'./logs/transformers/llama-2-7b/code/Lora_adapter/method_{config["method"]}/optimizer_{config["optimizer"]}/lr_{config["learning_rate"]}')
          tokenizer.save_pretrained(f'./logs/transformers/llama-2-7b/code/Lora_adapter/method_{config["method"]}/optimizer_{config["optimizer"]}/lr_{config["learning_rate"]}')
          
      wandb.finish()

if __name__ == "__main__":
    main()
