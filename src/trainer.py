import os
from typing import Any, List
import gc
import torch
import wandb
import yaml
import glob
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoTokenizer, GPT2Config, GPT2LMHeadModel,
                          get_scheduler)
from model.probe import ProbingOutput, LensProbingGPT2, NaturalProbingGPT2
from model.probe_vocab import VocabProbingGPT2
from tokenizer.wordlevel_tokenizer import TrainableWordTokenizer
from utils import step_num, get_files_sorted, checkpoint_path_to_model, tokenize_function, prepare_dataloader


seed = 42
AdamW = torch.optim.AdamW
BATCH_SIZE = 8
CHECKPOINTS_DIR = '/u501/x25luo/codebase/trabank-dev/model/childes_warmup_s42_shuffled/'
LEARNING_RATE = 5e-4
WANDB_LOG_EVERY = 1
PROJ_NAME = 'vocab_lens_ce'
effective_epochs = 4


def save_checkpoint(model: torch.nn.Module, optimizer: AdamW, scheduler: Any, epoch: int, epoch_step: int, global_step: int, layer_num:str):
    """Save a checkpoint with the model, optimizer, scheduler, and dataset state."""
    PROBE_CHECKPOINT_DIR = f'/u501/x25luo/codebase/probingLM/ckpt/vocab_len_childes_s{seed}_layer{layer_num}_ce'
    os.makedirs(PROBE_CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(PROBE_CHECKPOINT_DIR, f'checkpoint_{epoch}_{global_step}.pt')
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

def main(probe_layer:list, batch_size:int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = TrainableWordTokenizer(vocab_file='/u501/x25luo/codebase/probingLM/src/tokenizer/vocab.json')
    files_sorted = get_files_sorted(CHECKPOINTS_DIR)  # will be overrided
    model = checkpoint_path_to_model(files_sorted[-1], tokenizer, device)
    model.eval()
    probe_model = VocabProbingGPT2(model, tokenizer, probing_layers=probe_layer, loss_type="ce", device=device)
    probe_model.to(device)
    
    # dataset = load_dataset('Seed42Lab/childes-pretrain')['train']
    dataset = load_dataset("parquet", data_files={'train': '/u501/x25luo/codebase/probingLM/src/train-00000-of-00001.parquet'})['train']
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = prepare_dataloader(tokenized_dataset, batch_size, seed)
    
    optimizer = AdamW(probe_model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(
        'linear', optimizer=optimizer, num_warmup_steps=1000, num_training_steps=step_num(effective_epochs, dataset, batch_size)
    )
    wandb.init(project=PROJ_NAME, name=f'vocablens-seed42-layer_10_new', resume='allow')

    global_step = 0
    for epoch in range(effective_epochs):
        epoch_step = 0
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{effective_epochs}')

        for batch_no, batch in enumerate(progress_bar):
            vocab_size = model.config.vocab_size

            assert batch['input_ids'].max() < vocab_size, f"Error: Input ID = {batch['input_ids'].max()} exceeds vocab size={vocab_size} on {global_step}"

            batch = {key: value.to(device) for key, value in batch.items()}

            # Forward pass
            try:
                outputs = probe_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
                loss = outputs.total_loss
    
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                total_norm = 0.0
                for p in probe_model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)  # L2 norm
                        total_norm += param_norm.item() ** 2

                total_norm = total_norm ** 0.5  # L2 norm of all gradients combined
                torch.nn.utils.clip_grad_norm_(probe_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # Log loss
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

                # Increment global block number
                global_step += 1
                epoch_step += 1
                if global_step % WANDB_LOG_EVERY == 0:
                    wandb.log({'batch_loss': loss.item(), 'learning_rate': optimizer.param_groups[0]['lr'], 'epoch': epoch + 1, 'step_in_epoch': epoch_step, 'grad_norm': total_norm})

            except Exception as e:
                print(f'error: {e}')
                raise e

        gc.collect()
        torch.cuda.empty_cache()

        print(f'Epoch {epoch + 1} completed. Average loss: {epoch_loss / epoch_step}')
        save_checkpoint(probe_model, optimizer, scheduler, epoch, epoch_step, global_step, probe_layer[0])
        epoch_step = 0
    wandb.finish()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='analyze attention flow in GPT-2')
    parser.add_argument('--probe_layer', type=int, default=11)  # 0-11
    parser.add_argument('--batch_size', type=int, default=8)  # 0-11
    args = parser.parse_args()

    main([args.probe_layer], args.batch_size)