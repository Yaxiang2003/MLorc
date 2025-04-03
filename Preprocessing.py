import typing as tp
import functools
import os
import pickle
from typing import Any, Callable, Dict
import os

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from huggingface_hub import login

def cache_to_disk(root_datadir):
    def decorator_cache(func):
        @functools.wraps(func)
        def wrapper_cache(*args, **kwargs):
            if not os.path.exists(root_datadir):
                os.makedirs(root_datadir)

            func_name = func.__name__.replace("/", "")
            cache_file = os.path.join(root_datadir, f"{func_name}.pkl")

            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            return result

        return wrapper_cache

    return decorator_cache

def preprocess(
    tokenizer: AutoTokenizer,
    input_text: str,
    target_text: str,
    tokenizer_kwawgs: Dict[str, Any] = None,
):
    """
    standard preprocess function for dataset.
    Preprocesses input and target text data using a tokenizer object and returns a dictionary of model inputs.

    Args:
        tokenizer: An instance of a tokenizer class used to preprocess text data.
        input_text (str): A string containing the input text data to be tokenized.
        target_text (str, optional): A string containing the target text data to be tokenized. If None, no target data is returned.

    Returns:
        A dictionary of model inputs containing the tokenized input and output data along with the modified labels tensor.
    """
    if tokenizer_kwawgs is None:
        tokenizer_kwawgs = {}
    model_inputs = tokenizer(input_text, **tokenizer_kwawgs)
    if target_text is not None:
        labels = tokenizer(target_text, **tokenizer_kwawgs)
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
    return model_inputs


class DatasetPreprocessor:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        tokenizer_kwargs: Dict[str, Any] = None,
    ):
        """
        Initializes an instance of the datasets_preprocess class with a tokenizer object.

        Args:
            tokenizer: An instance of a tokenizer class used to preprocess text data.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

template_wo_input = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''

class MetaMathQA100k_Preprocessor(DatasetPreprocessor):
    # [TODO]

    def __call__(self, example):
        if isinstance(example["x"], str):
            # not batched
#             input_text, target_text = self.preprocess(
#                 example["instruction"], example["output"]
#             )
            raise NotImplementedError

        else:
            combined_text = [(x + " " + y + self.tokenizer.eos_token) for (x, y) in zip(example["x"], example["y"])]
            encodings = self.tokenizer(combined_text, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)

            labels = encodings["input_ids"].clone()
            input_text_length = [
                len(self.tokenizer(example["x"][i], return_tensors="pt")["input_ids"][0])
                for i in range(len(example["x"]))
            ]
            for i, l in enumerate(input_text_length):
                labels[i, :l] = -100
            labels[encodings["attention_mask"] == 0] = -100

            results = {
                "input_ids": encodings["input_ids"],
#                 "attention_mask": encodings["input_ids"].ne(self.tokenizer.pad_token_id),
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
            }

            return results

@cache_to_disk("data_cache")
def load_meta_math(max_tokens=1024):

    dataset = load_dataset('meta-math/MetaMathQA')['train']
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    def preprocess(data):
        return {
            "x": template_wo_input.format(
                instruction=data['query']
            ),
            "y": data["response"]
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=22000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens or "GSM" not in sample["type"]:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < 20000:  # First 100,000 samples for training
            train_samples.append(processed_sample)
        elif 20000 <= count < 22000:  # Next 10,000 samples for evaluation
            eval_samples.append(processed_sample)
        elif count >= 22000:  # Stop processing after collecting enough samples
            break
        count += 1

    # convert to hf dataset
    train_samples = Dataset.from_list(train_samples)
    eval_samples = Dataset.from_list(eval_samples)
    datasets = DatasetDict({
        "train": train_samples,
        "eval": eval_samples,
    })

    return datasets

class CodeFeedback100k_Preprocessor(DatasetPreprocessor):

    def __call__(self, example):
        if isinstance(example["x"], str):
            # not batched
            raise NotImplementedError
    
        else:
            combined_text = [(x + " " + y + self.tokenizer.eos_token) for (x, y) in zip(example["x"], example["y"])]
            encodings = self.tokenizer(combined_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)

            labels = encodings["input_ids"].clone()
            input_text_length = [
                len(self.tokenizer(example["x"][i], return_tensors="pt")["input_ids"][0])
                for i in range(len(example["x"]))
            ]
            for i, l in enumerate(input_text_length):
                labels[i, :l] = -100
            labels[encodings["attention_mask"] == 0] = -100
            
            results = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
            }

            return results

@cache_to_disk("data_cache")
def load_codefeedback(max_tokens=1024):
    dataset = dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction",  split="train")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    def preprocess(data):
        y = data['answer']
        y = "```".join(y.split("```")[:2]) + "```" # only keep the first code block
        return {
            "x": template_wo_input.format(
                instruction=data['query']
            ),
            "y": y,
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=22000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if "```" not in sample['answer']:
            continue
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < 22000:
            train_samples.append(processed_sample)
        elif 20000 <= count < 22000:
            eval_samples.append(processed_sample)
        elif count >= 22000:  # Stop processing after collecting enough samples
            break
        count += 1
        
    # convert to hf dataset
    train_samples = Dataset.from_list(train_samples)
    eval_samples = Dataset.from_list(eval_samples)
    datasets = DatasetDict({
        "train": train_samples,
        "eval": eval_samples,
    })
    
    return datasets
