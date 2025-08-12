'''Train native lens on CHILDS.
Use `torch.amp`. Refer to https://docs.pytorch.org/docs/stable/notes/amp_examples.html for details.'''

import os
from huggingface_hub import HfApi, HfFolder
from typing import Any, List
import gc
import torch
from torch.cuda.amp import autocast, GradScaler
import wandb
import yaml
import numpy as np
import random
import math
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (GPT2Config, GPT2LMHeadModel, get_scheduler)
from model.probe_vocab import VocabProbingGPT2
from tokenizer.wordlevel_tokenizer import TrainableWordTokenizer
from utils import tokenize_function, prepare_dataloader, get_shuffle_indices, watch_memory


def epoch_num(dataset: Dataset):
    """Determine epoch num with dataset."""
    return CYCLE_VALUE if CYCLE_MODE == 'epochs' else math.ceil(CYCLE_VALUE / math.ceil(len(dataset) / BATCH_SIZE))


def step_num(dataset: Dataset):
    """Determine step size with dataset."""
    return CYCLE_VALUE if CYCLE_MODE == 'steps' else math.ceil(len(dataset) / BATCH_SIZE) * CYCLE_VALUE


def load_checkpoint(checkpoint_path: int, dataset: Dataset, tokenizer):
    """Load a checkpoint and restore the training state."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restore model and wrap in DataParallel if needed
    base_model = GPT2LMHeadModel(config=GPT2Config())
    base_model.resize_token_embeddings(len(tokenizer))
    probe_model = VocabProbingGPT2(base_model, tokenizer, probing_layers=checkpoint['probing_layers'], loss_type="ce", device=device, add_layernorm=True)
    probe_model.load_state_dict(checkpoint['model_state_dict'])

    # Wrap in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1 and torch.cuda.is_available():
        print(f'Using {torch.cuda.device_count()} GPUs for resumed model')
        probe_model = torch.nn.DataParallel(probe_model)
    probe_model.to(device)

    optimizer = torch.optim.AdamW(probe_model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = get_scheduler(
        'linear', optimizer=optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=step_num(dataset)
    )
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Recompute unused portion of the dataset
    full_indices = get_shuffle_indices(len(dataset), SEED)
    start_index = checkpoint['epoch_step'] * BATCH_SIZE
    remaining_indices = full_indices[start_index:]
    unused_subset = dataset.select(remaining_indices)
    print(f'now size: {len(unused_subset)} instead of full {len(dataset)}')
    dataloader = DataLoader(unused_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    metadata = {
        'epoch': checkpoint['epoch'],
        'epoch_step': checkpoint['epoch_step'],
        'global_step': checkpoint['global_step'],
        'rng_state': checkpoint['rng_state'],
        'cuda_rng_state': checkpoint['cuda_rng_state']
    }
    return probe_model, optimizer, scheduler, dataloader, metadata


def resume_or_initialize_backbone(tokenizer, tokenized_dataset, dataset, probing_layer:list):
    # resume from last checkpoint
    try: 
        pt_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
        latest_checkpoint = sorted(pt_files, key=lambda x: int(x.split('_')[-1].replace('.pt', '')))[-1]
        print(f'Resuming from checkpoint: {latest_checkpoint}')
        checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
        probe_model, optimizer, scheduler, dataloader, metadata = load_checkpoint(checkpoint_path, tokenized_dataset, tokenizer)

        start_epoch = metadata['epoch']
        global_block_no = metadata['global_step']
        epoch_step = metadata['epoch_step']
        torch.set_rng_state(metadata['rng_state'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([it.cpu() for it in metadata['cuda_rng_state']])

    except:
        # load base model
        print('starting training from scratch')
        base_model = GPT2LMHeadModel(config=GPT2Config())
        base_model.resize_token_embeddings(len(tokenizer))

        # wrap up `probe model`
        probe_model = VocabProbingGPT2(base_model, tokenizer, probing_layers=probing_layer, loss_type="ce",device=device, add_layernorm=True)
        
        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            print(f'Using {torch.cuda.device_count()} GPUs with DataParallel')
            probe_model = torch.nn.DataParallel(probe_model)
        probe_model.to(device)

        optimizer = torch.optim.AdamW(probe_model.parameters(), lr=LEARNING_RATE)
        scheduler = get_scheduler(
            'linear', optimizer=optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=step_num(dataset)
        )
        dataloader = prepare_dataloader(tokenized_dataset, BATCH_SIZE, seed=SEED)

        start_epoch = 0
        global_block_no = 0
        epoch_step = 0

    return probe_model, optimizer, scheduler, dataloader, start_epoch, global_block_no, epoch_step


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.AdamW, scheduler: Any, epoch: int, epoch_step: int, global_step: int, probing_layer:list):
    """Save a checkpoint with the model, optimizer, scheduler, and dataset state."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(OUTPUT_DIR, f'checkpoint_{epoch}_{global_step}.pt')
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'model_state_dict': model_state_dict,
        'probing_layers': probing_layer,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'epoch_step': epoch_step,
        'global_step': global_step,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all()
    }, checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')
    
    try:
        hf_api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=f'checkpoint_{epoch}_{global_step}.pt',
            repo_id='Luoxiaoxi/GPT2-native-lens-CHILDS-seed42-bf16',
            repo_type="model"
        )
        print(f'Successfully uploaded {checkpoint_path} to HF')
    except Exception as e:
        print(f"Failed to upload {checkpoint_path} to Hugging Face Hub: {e}")


def init_training(probe_layers: List[int]):
    # load and preprocess dataset
    dataset = load_dataset("parquet", data_files={'train': TRAINING_DATA_PATH})['train']
    tokenizer = TrainableWordTokenizer(vocab_file=VOCAB_FILE)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    # initialize model and retraining situation
    probe_model, optimizer, scheduler, dataloader, start_epoch, global_step, epoch_step = resume_or_initialize_backbone(tokenizer, tokenized_dataset, dataset, probe_layers)
    return dataset, tokenizer, tokenized_dataset, probe_model, optimizer, scheduler, dataloader, start_epoch, global_step, epoch_step


def main(probe_layer:list):
    # initialize
    print('GPU number: ', torch.cuda.device_count())
    dataset, tokenizer, tokenized_dataset, probe_model, optimizer, scheduler, dataloader, start_epoch, global_step, epoch_step = init_training(probe_layer)
    
    wandb.init(project=PROJ_NAME, name=f'try parallel', resume='allow')
    scaler = GradScaler()
    effective_epochs = epoch_num(dataset)

    for epoch in range(start_epoch, effective_epochs):
        print(f'Current Epoch {epoch} ...', flush=True)
        if global_step == 0:
            save_checkpoint(probe_model, optimizer, scheduler, epoch, epoch_step, global_step, probe_layer)
        epoch_loss = 0
        # progress_bar = dataloader
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{effective_epochs}')

        for batch_no, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16): # Forward pass
                outputs = probe_model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                labels=batch['input_ids'])
                loss = outputs['total_loss']
                loss = loss.mean() if torch.cuda.device_count() > 1 else loss
            
            # loss.backward()
            scaler.scale(loss).backward()
            
            total_norm = 0.0
            for p in probe_model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5  # L2 norm of all gradients combined
            
            # Gradient clipping: Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(probe_model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

            # Increment global block number
            global_step += 1
            epoch_step += 1
            if global_step % WANDB_LOG_EVERY == 0:
                wandb.log({'batch_total_loss': loss.item(), 
                            'batch_gpt2_loss': outputs['loss_main'].mean().item(),
                            'batch_lens_loss': outputs['total_probe_loss'].mean().item(),
                            'learning_rate': optimizer.param_groups[0]['lr'], 
                            'epoch': epoch + 1, 
                            'step_in_epoch': epoch_step, 
                            'grad_norm': total_norm})
            
            # Save checkpoint
            if global_step % CHECKPOINT_INTERVAL == 0 or global_step in [500, 1500, 2500]:
                save_checkpoint(probe_model, optimizer, scheduler, epoch, epoch_step, global_step, probe_layer)
            
            # stop if it's 'steps' mode, at last epoch, and enough steps are trained (no need to finish the epoch)
            if CYCLE_MODE == 'steps' and epoch == effective_epochs - 1 and global_step == CYCLE_VALUE:
                print('Enough steps are trained.')
                print(f'Epoch {epoch + 1} completed.')
                break

            gc.collect()
            torch.cuda.empty_cache()

        print(f'Epoch {epoch + 1} completed. Average loss: {epoch_loss / epoch_step}')
        epoch_step = 0

    # Save the final model
    save_checkpoint(probe_model, optimizer, scheduler, epoch, epoch_step, global_step, probe_layer)
    # save_model_safely(probe_model, OUTPUT_DIR)
    print(f'Model saved to {OUTPUT_DIR}. Successfully finished!')
    wandb.finish()
    
    watch_memory()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    yaml_path = 'src/template_CHILDS.yaml'
    with open(yaml_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)

    SEED = hyperparameters['training']['seed']
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if using multi-GPU

    # pull all global VARs
    CHECKPOINT_DIR = hyperparameters['training']['checkpoint_dir']
    OUTPUT_DIR = hyperparameters['training']['output_dir']
    BATCH_SIZE = hyperparameters['training']['batch_size_per_gpu']
    MAX_LENGTH = hyperparameters['model']['tokenizer']['model_max_length']
    CYCLE_MODE = hyperparameters['training']['training_cycle_unit']
    CYCLE_VALUE = hyperparameters['training']['training_cycle_value']
    LEARNING_RATE = hyperparameters['training']['lr']
    TRAINING_DATA_PATH = hyperparameters['training']['path']
    WARMUP_STEPS = hyperparameters['training']['warmup_steps']
    CHECKPOINT_INTERVAL = hyperparameters['training']['checkpoint_every']
    WANDB_EXP_NAME = hyperparameters['training']['wandb_exp_name']
    WANDB_LOG_EVERY = hyperparameters['training']['wandb_log_every']
    TOKENIZER_TYPE = hyperparameters['model']['tokenizer']['tokenizer_type']
    VOCAB_FILE = hyperparameters['model']['tokenizer']['vocab_file']
    PROJ_NAME = hyperparameters['training']['wandb_project']

    hf_api = HfApi()
    
    main(probe_layer = [i for i in range(11)])