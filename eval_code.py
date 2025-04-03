
import os
import sys
import re
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from human_eval.data import write_jsonl, read_problems

from huggingface_hub import login, notebook_login
from tqdm import tqdm
from typing import List
from einops import rearrange
from datasets import load_dataset
import accelerate
import transformers
from transformers import default_data_collator
import copy
from fractions import Fraction

config = {
    "learning_rate": 5e-5,
    "optimizer": "MLorc_Lion",
}

def post_process(text):
    text = text.replace("```", "")
    text = text.replace("\t", "    ")
    text = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', text, flags=re.DOTALL)
    text = "\n".join([ll.rstrip() for ll in text.splitlines() if ll.strip()])
    lines = text.split("\n")
    spaces_for_each_line = []
    for line in lines:
        match = re.match(r'^( *)', line)
        if match:
            leading_spaces = len(match.group(1))
            spaces_for_each_line.append(leading_spaces)
    try:
        def_line = [i for i, line in enumerate(lines) if "def" in line][0]
        def_line_space = spaces_for_each_line[def_line]
    except:
        print("No def line found")
        print(text)
        def_line_space = 0
    rank_unique_spaces = sorted(list(set(spaces_for_each_line)))
    indentation_level = {}
    i = 0
    for space in rank_unique_spaces:
        if space <= def_line_space:
            indentation_level[space] = 0
        else:
            i += 1
            indentation_level[space] = i
    new_lines = []
    for line, space in zip(lines, spaces_for_each_line):
        new_lines.append("    " * indentation_level[space] + line.lstrip())
    return "\n".join(new_lines)

ALPACA_PREFIX_TEMPLATE_MD = """Below is an instruction that describes a task.\n Write a response that appropriately completes the request.

### Instruction:
Complete the following Python code: 
Notes: respond with the entire complete function definition
do not add any comments, be as concise in your code as possible
use only built-in libraries, assume no additional imports other than those provided (if any)
use `    ` (4 spaces) for each level of indentation

code:
```python
{PROMPT}
```

### Response:
```python
"""
    

def split_dataset(dataset, rank, world_size):
    total_size = len(dataset)
    per_process_size = math.ceil(total_size / world_size)
    start_index = rank * per_process_size
    end_index = min(start_index + per_process_size, total_size)
    subset = torch.utils.data.Subset(dataset, list(range(start_index, end_index)))
    return subset

def gather_from_all_processes(data):
    """Gather data from all processes and concatenate."""
    gathered_data = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_data, data)
    # Flatten the list of lists
    return [item for sublist in gathered_data for item in sublist]

def compute_accuracy(predictions, references):
    """Compute accuracy by comparing predictions and references."""
    correct = sum([pred == ref for pred, ref in zip(predictions, references)])
    return correct / len(predictions)


@torch.inference_mode()
def main():

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    print("Environment initialized successfully!Number of gpu is:", torch.cuda.device_count())

    # Step 1: load model
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, max_length=1024, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}, use_auth_token=True)
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name, padding_side="left")

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    model.to(local_rank)
    model.eval()

    # Step 2: load dataset
    dataset = read_problems()
    dataset = [v for (k, v) in dataset.items()]
    
    dataset = split_dataset(dataset, local_rank, world_size)
    dataset = Dataset.from_list(dataset)
    
    def preprocess(examples):
        task_ids = [int(task_id.split("/")[-1]) for task_id in examples["task_id"]]
        input_texts = [(ALPACA_PREFIX_TEMPLATE_MD.format(PROMPT=prompt) + " ") for prompt in examples["prompt"]]
        
        encodings = tokenizer(input_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=768)
        
        results = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "task_ids": task_ids,
        }
        return results

    dataset = dataset.map(
        preprocess,
        batched=True,
        batch_size=1000,
        num_proc=1,
        desc="Running tokenizer on dataset",
        remove_columns=["entry_point", "canonical_solution", "test", "prompt"],
    )
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=default_data_collator)

    # Step 3: evaluation on test dataset
    all_predictions = []
    all_references = []

    model.eval()
    with torch.no_grad():
        i = 0
        t = tqdm(dataloader) if dist.get_rank() == 0 else dataloader
        for batch in t:
            i = i+1
            outputs = model.generate(
                batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=512,
                eos_token_id=tokenizer.eos_token_id,
                top_p=0.95,
                temperature=0.8,
            )
            predictions = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            pred = []
            for prediction in predictions:
                pred.append(extract_num(prediction))
            all_predictions.extend(pred)
            all_references.extend(batch["labels"].tolist())


    all_predictions = gather_from_all_processes(all_predictions)
    all_references = gather_from_all_processes(all_references)

    if dist.get_rank() == 0:
        accuracy = compute_accuracy(all_predictions, all_references)
        print(f"Test samples {len(all_predictions)}")
        print(f"Test samples {len(all_references)}")
        print(f"Final Accuracy: {100. * accuracy}")
        print("optimizer:", config["optimizer"])
        print("lr:", config["learning_rate"])

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
