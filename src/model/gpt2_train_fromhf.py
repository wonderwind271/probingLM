"""The gpt2_train_fromhf.py can train a gpt-2 model with huggingface dataset specified. It can store checkpoints every CHECKPOINT_INTERVAL (default 100) batches of data. The checkpoint includes a 'block_no' that tracks which training data we have already covered. load_checkpoint function can recover the model as well as dataloader with only untrained parts.

Dataset is not shuffled in this version of implementation.
"""
import math
import os
import random
from typing import Any, List

import numpy as np
import torch
import wandb
import yaml
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoTokenizer, GPT2Config, GPT2LMHeadModel,
                          get_scheduler)
from tokenizer.wordlevel_tokenizer import TrainableWordTokenizer

yaml_path = 'template_CHILDS.yaml'
with open(yaml_path, 'r') as file:
    hyperparameters = yaml.safe_load(file)

# pull all global VARs
CHECKPOINT_DIR = hyperparameters['training']['checkpoint_dir']
OUTPUT_DIR = hyperparameters['training']['output_dir']
BATCH_SIZE = hyperparameters['training']['batch_size_per_gpu']
MAX_LENGTH = hyperparameters['model']['tokenizer']['model_max_length']
CYCLE_MODE = hyperparameters['training']['training_cycle_unit']
CYCLE_VALUE = hyperparameters['training']['training_cycle_value']
LEARNING_RATE = hyperparameters['training']['lr']
WARMUP_STEPS = hyperparameters['training']['warmup_steps']
CHECKPOINT_INTERVAL = hyperparameters['training']['checkpoint_every']
DATASET_PATH = hyperparameters['training']['path']
IS_TRAIN_ALL = hyperparameters['training']['is_train_all']
TRAIN_SPLIT = hyperparameters['training']['train_split']
WANDB_PROJ = hyperparameters['training']['wandb_project']
WANDB_EXP_NAME = hyperparameters['training']['wandb_exp_name']
WANDB_LOG_EVERY = hyperparameters['training']['wandb_log_every']
TOKENIZER_TYPE = hyperparameters['model']['tokenizer']['tokenizer_type']
VOCAB_FILE = hyperparameters['model']['tokenizer']['vocab_file']


def step_num(dataset: Dataset):
    """Determine step size with dataset."""
    return CYCLE_VALUE if CYCLE_MODE == 'steps' else math.ceil(len(dataset) / BATCH_SIZE) * CYCLE_VALUE


def epoch_num(dataset: Dataset):
    """Determine epoch num with dataset."""
    return CYCLE_VALUE if CYCLE_MODE == 'epochs' else math.ceil(CYCLE_VALUE / math.ceil(len(dataset) / BATCH_SIZE))


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


def save_checkpoint(model: torch.nn.Module, optimizer: AdamW, scheduler: Any, epoch: int, epoch_step: int, global_step: int):
    """Save a checkpoint with the model, optimizer, scheduler, and dataset state."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{epoch}_{global_step}.pt')
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'epoch_step': epoch_step,
        'global_step': global_step,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all()
    }, checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')


def load_checkpoint(checkpoint_path: int, dataset: Dataset, tokenizer):
    """Load a checkpoint and restore the training state."""
    checkpoint = torch.load(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Restore model and wrap in DataParallel if needed
    model = GPT2LMHeadModel(config=GPT2Config())
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Wrap in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for resumed model')
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = get_scheduler(
        'linear', optimizer=optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=step_num(dataset)
    )
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Recompute unused portion of the dataset
    full_indices = get_shuffle_indices(len(dataset), seed)
    start_index = checkpoint['epoch_step'] * BATCH_SIZE
    remaining_indices = full_indices[start_index:]
    unused_subset = dataset.select(remaining_indices)

    print(f'now size: {len(unused_subset)} instead of full {len(dataset)}')
    dataloader = DataLoader(unused_subset, batch_size=BATCH_SIZE, shuffle=False)

    metadata = {
        'epoch': checkpoint['epoch'],
        'epoch_step': checkpoint['epoch_step'],
        'global_step': checkpoint['global_step'],
        'rng_state': checkpoint['rng_state'],
        'cuda_rng_state': checkpoint['cuda_rng_state']
    }
    return model, optimizer, scheduler, dataloader, metadata


def tokenize_function(example, tokenizer):
    """Tokenize the dataset examples."""
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)


def prepare_dataloader(dataset: Dataset, batch_size: int):
    """Prepare the DataLoader for the unused portion of the dataset."""
    g = torch.Generator()
    g.manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
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


def train(tokenizer_type: str = 'gpt2'):
    """Main training loop with checkpointing and dataset tracking. tokenizer_type is gpt2 or wordlevel."""
    # Load the dataset
    dataset = split_dataset(load_merge_dataset(DATASET_PATH, load_all=IS_TRAIN_ALL, load_split=TRAIN_SPLIT))
    # dataset = load_from_disk('/home/shuyuwu/trabank-dev/data/childes_dataset/')
    print('Dataset example:')
    print(dataset['text'][0])
    if tokenizer_type == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    else:
        tokenizer = TrainableWordTokenizer(VOCAB_FILE)

    tokenizer.pad_token = tokenizer.eos_token
    wandb.init(project=WANDB_PROJ, name=WANDB_EXP_NAME, resume='allow')
    # Tokenize the dataset
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    # tokenized_dataset.save_to_disk("/scratch/chaijy_root/chaijy2/shuyuwu/experiments/childes_warmup_tokenized/")
    # tokenized_dataset = load_from_disk("/scratch/chaijy_root/chaijy2/shuyuwu/experiments/data2/")
    # Check if resuming from a checkpoint
    if os.path.exists(CHECKPOINT_DIR):
        latest_checkpoint = sorted(os.listdir(CHECKPOINT_DIR), key=lambda x: int(x.split('_')[-1][:-3]))[-1]
        print(f'Resuming from checkpoint: {latest_checkpoint}')
        checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
        model, optimizer, scheduler, dataloader, metadata = load_checkpoint(checkpoint_path, tokenized_dataset, tokenizer)

        start_epoch = metadata['epoch']
        global_block_no = metadata['global_step']
        epoch_step = metadata['epoch_step']
        torch.set_rng_state(metadata['rng_state'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(metadata['cuda_rng_state'])
    else:
        print('Starting training from scratch.')
        model = GPT2LMHeadModel(config=GPT2Config())
        model.resize_token_embeddings(len(tokenizer))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs with DataParallel')
            model = torch.nn.DataParallel(model)

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = get_scheduler(
            'linear', optimizer=optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=step_num(dataset)
        )
        dataloader = prepare_dataloader(tokenized_dataset, BATCH_SIZE)
        start_epoch = 0
        global_block_no = 0
        epoch_step = 0

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    complete_dataloader = prepare_dataloader(tokenized_dataset, BATCH_SIZE)
    # Training loop
    model.train()

    effective_epochs = epoch_num(dataset)
    for epoch in range(start_epoch, effective_epochs):
        # if global_block_no == 0:
        #     save_checkpoint(model, optimizer, scheduler, epoch, epoch_step, global_block_no)
        epoch_loss = 0
        if epoch == start_epoch:
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{effective_epochs}')
        else:
            progress_bar = tqdm(complete_dataloader, desc=f'Epoch {epoch + 1}/{effective_epochs}')

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
                loss = outputs.loss.mean()

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

                # wandb
                if global_block_no % WANDB_LOG_EVERY == 0:
                    wandb.log({'batch_loss': loss.item(), 'learning_rate': optimizer.param_groups[0]['lr'], 'epoch': epoch + 1, 'step_in_epoch': epoch_step, 'grad_norm': total_norm})

                # Save checkpoint
                if global_block_no % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(model, optimizer, scheduler, epoch, epoch_step, global_block_no)
            except Exception as e:
                print(f'error: {e}')
                raise e

            # stop if it's 'steps' mode, at last epoch, and enough steps are trained (no need to finish the epoch)
            if CYCLE_MODE == 'steps' and epoch == effective_epochs - 1 and global_block_no == CYCLE_VALUE:
                print('Enough steps are trained.')
                print(f'Epoch {epoch + 1} completed.')
                break

        print(f'Epoch {epoch + 1} completed. Average loss: {epoch_loss / epoch_step}')
        epoch_step = 0

    # Save the final model
    save_model_safely(model, OUTPUT_DIR)
    print(f'Model saved to {OUTPUT_DIR}.')
    wandb.finish()


def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    
if __name__ == '__main__':
    SEED = hyperparameters['training']['seed']
    set_seed(SEED)
    
    train(TOKENIZER_TYPE)
