import os
import argparse
import logging
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from datasets import DatasetDict, load_dataset
import transformers
from transformers import TrainingArguments, Trainer
from transformers import default_data_collator
from huggingface_hub import login, notebook_login
from lion_pytorch import Lion

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

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = RSVD_CM_AdamW(model.parameters(), lr=2e-5, rank=8)
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

def main():
  local_rank = int(os.getenv("LOCAL_RANK", "0"))
  world_size = int(os.getenv("WORLD_SIZE", "1"))
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
  train_args = TrainingArguments(
        output_dir="./llama-2-7b-metamathqa100k",
        run_name = "llama2_7b_math_experiment",
        logging_dir="./llama-2-7b-metamathqa100k_logs",
        do_train=True,
        num_train_epochs=2,
        per_device_train_batch_size=50,
        gradient_accumulation_steps=4,
        #optim="adamw_torch",
        logging_steps=1,
        bf16=True,
        #learning_rate=2e-5,
        weight_decay=0, # No weight decay
        warmup_ratio=0.03,
        #lr_scheduler_type="cosine",
        report_to="wandb" if local_rank == 0 else None,
        label_names=[
            "labels"
        ],
        ddp_find_unused_parameters=False,
        do_eval=True,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=-1,
        save_strategy="no",
        deepspeed="./deepspeed_zero2.json" if world_size > 1 else None,
    )


  trainer = CustomTrainer(
      model=model,
      args=train_args,
      train_dataset=datasets["train"],
      eval_dataset=datasets["eval"],
      data_collator=default_data_collator,
      optimizers=(my_optimizer, my_lr_scheduler),
  )
  trainer.train()

  if local_rank == 0:
    model.save_pretrained(f'./logs/transformers/llama-2-7b/math/Adam_RSVD_CM/2epoch')
    tokenizer.save_pretrained(f'./logs/transformers/llama-2-7b/math/Adam_RSVD_CM/2epoch')

    wandb.finish()

if __name__ == "__main__":
    main()
