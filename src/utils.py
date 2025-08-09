import math
import os
import random
from typing import Any, List

import numpy as np
import torch
import glob
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoTokenizer, GPT2Config, GPT2LMHeadModel,
                          get_scheduler)


def step_num(epoches: int, dataset: Dataset, batch_size:int):
    """Determine step size with dataset."""
    return math.ceil(len(dataset) / batch_size) * epoches


def split_text_into_chunks(text, chunk_size=512):
    """Split a string into chunks of at most `chunk_size` words."""
    words = text.split()
    # Generate chunks by slicing the list of words
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def get_shuffle_indices(N: int, torch_seed: int = 42) -> list:
    """Get the shuffle index for a seed."""
    g = torch.Generator().manual_seed(torch_seed)
    return torch.randperm(N, generator=g).tolist()


def split_dataset(dataset, chunk_size=512):
    """Split the dataset."""
    new_records = []
    new_index = 0

    for record in dataset:
        text_chunks = split_text_into_chunks(record['text'], chunk_size)
        for chunk in text_chunks:
            new_records.append({'index': new_index, 'text': chunk})
            new_index += 1

    new_dataset = Dataset.from_dict({'index': [r['index'] for r in new_records],
                                    'text': [r['text'] for r in new_records]})
    return new_dataset


def extract_step(fname):
    """Extracts the step number from a filename of the form: 'checkpoint_X_YYYY.pt' where X can be any integer index and YYYY is the step number."""
    basename = os.path.basename(fname)
    # e.g., 'checkpoint_0_150.pt' -> parts = ['checkpoint', '0', '150.pt']
    parts = basename.split('_')
    step_str = parts[-1].replace('.pt', '')
    return int(step_str)


def get_files_sorted(dir: str):
    """Generate a list of checkpoint in order, given the checkpoint dir."""
    c_pattern = os.path.join(dir, 'checkpoint_*.pt')
    c_files = glob.glob(c_pattern)
    c_files_sorted = sorted(c_files, key=extract_step)
    return c_files_sorted


def checkpoint_path_to_model(path, tokenizer, device):
    """Load checkpoint to model."""
    checkpoint = torch.load(path, map_location=device)
    model = GPT2LMHeadModel(config=GPT2Config())
    # NOTICE: only the following could work unders Xiaoxi's environment
    model.resize_token_embeddings(tokenizer.vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model



def tokenize_function(example, tokenizer):
    """Tokenize the dataset examples."""
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)


def prepare_dataloader(dataset: Dataset, batch_size: int, seed: int):
    """Prepare the DataLoader for the unused portion of the dataset."""
    g = torch.Generator()
    g.manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g, num_workers=2)
    return dataloader


def load_merge_dataset(path: str, load_all: bool = True, load_split: List[str] = []):
    """Load dataset, and merge specified or all splits into one single Dataset object."""
    dataset_dict = load_dataset(path)
    # return dataset_dict
    if load_all:
        merged_dataset = concatenate_datasets(list(dataset_dict.values()))  # Merge all splits into one Dataset
    else:
        missing_splits = [split for split in load_split if split not in dataset_dict]
        if missing_splits:
            raise ValueError(f'Specified splits not found in the dataset: {missing_splits}')
        merged_dataset = concatenate_datasets([dataset_dict[split] for split in load_split])  # Merge only the specified splits
    return merged_dataset


def save_model_safely(model, output_dir):
    """Save model with one or more GPU."""
    if hasattr(model, 'module'):  # i.e., it's wrapped in DataParallel
        model_to_save = model.module
    else:
        model_to_save = model
    model_to_save.save_pretrained(output_dir)