import math
import os
import random
from typing import Any, List

import numpy as np
import torch
import wandb
import yaml
import glob
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoTokenizer, GPT2Config, GPT2LMHeadModel,
                          get_scheduler)

from model.probe import ProbingOutput, LensProbingGPT2, NaturalProbingGPT2
from tokenizer.wordlevel_tokenizer import TrainableWordTokenizer

seed = 42
tokenizer = TrainableWordTokenizer(vocab_file='tokenizer/vocab.json')

def split_text_into_chunks(text, chunk_size=512):
    """Split a string into chunks of at most `chunk_size` words."""
    words = text.split()
    # Generate chunks by slicing the list of words
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def get_shuffle_indices(N: int, torch_seed: int = seed) -> list:
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

def checkpoint_path_to_model(path):
    """Load checkpoint to model."""
    checkpoint = torch.load(path)
    model = GPT2LMHeadModel(config=GPT2Config())
    model.resize_token_embeddings(len(tokenizer))
    # model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def tokenize_function(example, tokenizer):
    """Tokenize the dataset examples."""
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)


def prepare_dataloader(dataset: Dataset, batch_size: int):
    """Prepare the DataLoader for the unused portion of the dataset."""
    g = torch.Generator()
    g.manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
    return dataloader



CHECKPOINTS_DIR = '/scratch/chaijy_root/chaijy2/shuyuwu/experiments/checkpoints/childes_warmup_s42_shuffled/'
files_sorted = get_files_sorted(CHECKPOINTS_DIR)  # will be overrided
LEARNING_RATE = 5e-4

print(files_sorted[-1])
effective_epochs = 4

if __name__ == '__main__':
    model = checkpoint_path_to_model(files_sorted[-1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probe_model = LensProbingGPT2(model, tokenizer)
    dataset = load_dataset('wonderwind271/childes-pretrain')
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = prepare_dataloader(tokenized_dataset, 16)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(
        'linear', optimizer=optimizer, num_warmup_steps=1000, num_training_steps=4*len(dataset)
    )
    global_block_no = 0
    for epoch in range(effective_epochs):
        epoch_step = 0
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{effective_epochs}')

        for batch_no, batch in enumerate(progress_bar):
            vocab_size = model.config.vocab_size

            assert batch['input_ids'].max() < vocab_size, f"Error: Input ID = {batch['input_ids'].max()} exceeds vocab size={vocab_size} on {global_block_no}"

            if global_block_no <= -1:
                global_block_no += 1
                epoch_step += 1
                continue
            batch = {key: value.to(device) for key, value in batch.items()}

            # Forward pass
            try:
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
                loss = outputs.total_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)  # L2 norm
                        total_norm += param_norm.item() ** 2

                total_norm = total_norm ** 0.5  # L2 norm of all gradients combined
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                # Log loss
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

                # Increment global block number
                global_block_no += 1
                epoch_step += 1

            except Exception as e:
                print(f'error: {e}')
                raise e

        print(f'Epoch {epoch + 1} completed. Average loss: {epoch_loss / epoch_step}')
        epoch_step = 0
