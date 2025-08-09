#!/usr/bin/env python3
import os
from typing import Any, List
import gc
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import yaml
import numpy as np
import random
import math
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (GPT2Config, GPT2LMHeadModel, get_scheduler)
from model.probe_vocab import VocabProbingGPT2
from tokenizer.wordlevel_tokenizer import TrainableWordTokenizer
from utils import checkpoint_path_to_model, tokenize_function, prepare_dataloader, get_shuffle_indices, save_model_safely
from torch.cuda.amp import autocast, GradScaler

# -------------------
# UTILITIES
# -------------------
def strip_module_prefix(state_dict):
    """Remove 'module.' prefix from DDP checkpoints saved from model.module.state_dict()."""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    return new_state


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# -------------------
# TRAINING HELPERS
# -------------------
def epoch_num(dataset_len: int):
    """Determine epoch num with dataset length (per full dataset, not per replica)."""
    return CYCLE_VALUE if CYCLE_MODE == 'epochs' else math.ceil(CYCLE_VALUE / math.ceil(dataset_len / (BATCH_SIZE * WORLD_SIZE)))


def step_num(dataset_len: int):
    """Steps for scheduler: approximate total steps across all replicas."""
    if CYCLE_MODE == 'steps':
        return CYCLE_VALUE
    steps_per_epoch = math.ceil(dataset_len / (BATCH_SIZE * WORLD_SIZE))
    return steps_per_epoch * CYCLE_VALUE


def load_checkpoint(checkpoint_path: str, tokenizer, device):
    """Load checkpoint and return model_state, opt_state, scheduler_state, metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint['model_state_dict']
    model_state = strip_module_prefix(model_state)
    return {
        'model_state_dict': model_state,
        'probing_layers': checkpoint.get('probing_layers'),
        'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
        'scheduler_state_dict': checkpoint.get('scheduler_state_dict'),
        'epoch': checkpoint.get('epoch', 0),
        'epoch_step': checkpoint.get('epoch_step', 0),
        'global_step': checkpoint.get('global_step', 0),
        'rng_state': checkpoint.get('rng_state'),
        'cuda_rng_state': checkpoint.get('cuda_rng_state')
    }


# -------------------
# MODEL / CHECKPOINT
# -------------------
def save_checkpoint_if_main(model: torch.nn.Module, optimizer, scheduler: Any, epoch: int, epoch_step: int, global_step: int, probing_layer: list):
    if not is_main_process():
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(OUTPUT_DIR, f'checkpoint_{epoch}_{global_step}.pt')
    model_state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
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
    print(f'Checkpoint saved to {checkpoint_path}', flush=True)

# -------------------
# MAIN: resume or init
# -------------------
def resume_or_initialize_backbone(tokenizer, tokenized_dataset, dataset_len, probing_layer: list, device, local_rank):
    # If checkpoint exists, read latest (main proc decides)
    latest_checkpoint = None
    metadata = None

    if is_main_process() and os.path.exists(CHECKPOINT_DIR):
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].replace('.pt', '')))[-1]
            print(f'Resuming from checkpoint: {latest_checkpoint}', flush=True)
            latest_checkpoint = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
            metadata = load_checkpoint(latest_checkpoint, tokenizer, device)

    # synchronize so each rank gets info whether there is a checkpoint
    latest_checkpoint_flag = 1 if latest_checkpoint else 0
    latest_checkpoint_flag = torch.tensor(latest_checkpoint_flag, device=device)
    if dist.is_initialized():
        dist.broadcast(latest_checkpoint_flag, src=0)
    latest_exists = bool(latest_checkpoint_flag.item())

    if latest_exists and not metadata:
        # non-main ranks load checkpoint file path by waiting for the path string broadcast.
        # Simpler: let every rank load the same file (we ensure path exists and is accessible).
        metadata = load_checkpoint(latest_checkpoint, tokenizer, device)

    # build model
    base_model = GPT2LMHeadModel(config=GPT2Config())
    base_model.resize_token_embeddings(len(tokenizer))
    probe_model = VocabProbingGPT2(base_model, tokenizer, probing_layers=(metadata['probing_layers'] if metadata and metadata.get('probing_layers') else probing_layer), loss_type="ce")

    # load weights if checkpoint present
    if metadata and metadata.get('model_state_dict') is not None:
        state = metadata['model_state_dict']
        probe_model.load_state_dict(state, strict=False)

    probe_model.to(device)
    probe_model = DDP(probe_model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(probe_model.parameters(), lr=LEARNING_RATE)
    if metadata and metadata.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(metadata['optimizer_state_dict'])

    scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=step_num(dataset_len))
    if metadata and metadata.get('scheduler_state_dict') is not None:
        try:
            scheduler.load_state_dict(metadata['scheduler_state_dict'])
        except Exception:
            # scheduler state mismatch sometimes; ignore if cannot load
            if is_main_process():
                print("Warning: could not load scheduler state (mismatch). Starting fresh scheduler.", flush=True)

    # prepare dataloader with DistributedSampler
    sampler = DistributedSampler(tokenized_dataset, shuffle=True, seed=SEED)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    if metadata:
        start_epoch = metadata['epoch']
        global_step = metadata['global_step']
        epoch_step = metadata['epoch_step']
        # restore RNG states only on main then broadcast? We'll set per-rank deterministic seed below
    else:
        start_epoch = 0
        global_step = 0
        epoch_step = 0

    return probe_model, optimizer, scheduler, dataloader, start_epoch, global_step, epoch_step, sampler


# -------------------
# TRAINING LOOP
# -------------------
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # init process group if multiple GPUs requested
    if "WORLD_SIZE" in os.environ:
        global WORLD_SIZE
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    else:
        WORLD_SIZE = torch.cuda.device_count()
    # initialize distributed backend if multiple processes
    if WORLD_SIZE > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
    else:
        # no dist
        pass

    # per-process device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # set seeds per rank for reproducibility
    rank = dist.get_rank() if dist.is_initialized() else 0
    random.seed(SEED + rank)
    np.random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED + rank)
        torch.cuda.manual_seed_all(SEED + rank)

    # load dataset (each rank loads; acceptable for single-node)
    dataset = load_dataset("parquet", data_files={'train': TRAINING_DATA_PATH})['train']
    tokenizer = TrainableWordTokenizer(vocab_file=VOCAB_FILE)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataset_len = len(tokenized_dataset)

    # build model, optimizer, dataloader etc.
    probe_model, optimizer, scheduler, dataloader, start_epoch, global_step, epoch_step, sampler = resume_or_initialize_backbone(tokenizer, tokenized_dataset, dataset_len, probe_layers, device, local_rank)

    # only rank 0 initializes W&B and prints top-level logs
    if is_main_process():
        # os.environ["WANDB_MODE"] = "offline"  # safe default
        wandb.init(project=PROJ_NAME, name=f'try_ddp', resume='allow')

    # synchronize
    if dist.is_initialized():
        dist.barrier()

    effective_epochs = epoch_num(dataset_len)
    for epoch in range(start_epoch, effective_epochs):
        if dist.is_initialized():
            sampler.set_epoch(epoch)  # important for shuffling across epochs
        if is_main_process():
            print(f'Current Epoch {epoch} ...', flush=True)
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_no, batch in enumerate(dataloader):
            if is_main_process():
                print(f"[DEBUG][rank{rank}] global_step={global_step}, batch_no={batch_no}", flush=True)

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # Forward
            with autocast():
                outputs = probe_model(input_ids=batch['input_ids'],
                                    attention_mask=batch['attention_mask'],
                                    labels=batch['input_ids'])
                loss = outputs['total_loss']
                if loss.dim() > 0:
                    loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            # gradient clipping (DDP: gradients synchronized during backward)
            torch.nn.utils.clip_grad_norm_(probe_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1

            global_step += 1
            epoch_step += 1

            if is_main_process() and global_step % WANDB_LOG_EVERY == 0:
                wandb.log({
                    'batch_total_loss': loss.item(),
                    'batch_gpt2_loss': outputs.get('loss_main', torch.tensor(0.0)).mean().item() if 'loss_main' in outputs else 0.0,
                    'batch_lens_loss': outputs.get('total_probe_loss', torch.tensor(0.0)).mean().item() if 'total_probe_loss' in outputs else 0.0,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch + 1,
                    'step_in_epoch': epoch_step,
                })

            if is_main_process() and (global_step % CHECKPOINT_INTERVAL == 0 or global_step in [0, 150, 300]):
                save_checkpoint_if_main(probe_model, optimizer, scheduler, epoch, epoch_step, global_step, probe_layers)

            if CYCLE_MODE == 'steps' and epoch == effective_epochs - 1 and global_step == CYCLE_VALUE:
                if is_main_process():
                    print('Enough steps are trained. Breaking.', flush=True)
                break

            # housekeeping
            gc.collect()
            torch.cuda.empty_cache()

        # only main prints epoch summary
        if is_main_process():
            avg_loss = epoch_loss / max(1, epoch_steps)
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss}", flush=True)

        epoch_step = 0

    # finalize: save model only on main
    if is_main_process():
        save_model_safely(probe_model.module if isinstance(probe_model, DDP) else probe_model, OUTPUT_DIR)
        print(f"Model saved to {OUTPUT_DIR}.", flush=True)
        wandb.finish()

    # cleanup
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    # Load config
    yaml_path = 'src/template_CHILDS.yaml'
    with open(yaml_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)

    SEED = hyperparameters['training']['seed']
    # global variables from config
    CHECKPOINT_DIR = hyperparameters['training']['checkpoint_dir']
    OUTPUT_DIR = hyperparameters['training']['output_dir']
    BATCH_SIZE = hyperparameters['training']['batch_size_per_gpu']  # per GPU
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
    probe_layers = [i for i in range(11)]

    # dataloader workers tuning
    NUM_WORKERS = 2  # adjust according to sbatch --cpus-per-task
    # WORLD_SIZE will be set in main() from env by torchrun
    scaler = GradScaler()

    main()
