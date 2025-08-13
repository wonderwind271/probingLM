# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun --standalone --nproc_per_node=4 train_hf_trainer.py


from model.probe import ProbingOutput
from model.probe_vocab import VocabProbingGPT2
from tokenizer import TrainableWordTokenizer

import os
import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Config
)

# -----------------------------
# Model & Tokenizer
# -----------------------------
# If you already have `model` instantiated, comment out the next line and import/construct yours.
model_name = "gpt2"
base_model = GPT2LMHeadModel(config=GPT2Config())
tokenizer = TrainableWordTokenizer(vocab_file='/u501/x25luo/codebase/probingLM/src/tokenizer/vocab.json')
base_model.resize_token_embeddings(len(tokenizer))
# wrap up `probe model`
probing_layer = list(range(-1,11))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
probe_model = VocabProbingGPT2(base_model, tokenizer, probing_layers=probing_layer, loss_type="ce",device=device, add_layernorm=True)

# If you already have a tokenizer, comment this and use yours.
tokenizer = AutoTokenizer.from_pretrained(model_name)
# GPT-2 has no pad token; map pad -> eos for LM training.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Data
# -----------------------------
# Small, quick dataset for demo. Replace with your dataset when ready.
local_dataset_name = ""
raw = load_from_disk(local_dataset_name)
# Train/val splits are "train" and "validation" here.

# Simple tokenizer function
def tok_fn(examples):
    # Truncate to a reasonable length; adjust as needed
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized = raw.map(tok_fn, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -----------------------------
# Hyperparameters
# -----------------------------
per_device_bs = 8                 # what fits in VRAM
grad_accum = 2                    # 8 * 2 = 16 effective per GPU
max_steps = 20000                 # keep steps the same; accumulation doesn't change total optimizer steps
warmup_ratio = 0.05               # easier than counting steps manually
learning_rate = 5e-5              # adjust to your model/task
weight_decay = 0.05
max_grad_norm = 1.0
seed = 42

# Mixed precision—set bf16=True on newer GPUs (A100/H100), otherwise fp16=True
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+
fp16 = not use_bf16
bf16 = use_bf16

# -----------------------------
# TrainingArguments
# -----------------------------
args = TrainingArguments(
    output_dir="out",
    overwrite_output_dir=True,
    per_device_train_batch_size=per_device_bs,
    per_device_eval_batch_size=per_device_bs,
    gradient_accumulation_steps=grad_accum,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,                 # use max_steps to match your planned 20k optimizer updates
    warmup_ratio=warmup_ratio,           # counted in optimizer-step units
    lr_scheduler_type="linear",
    logging_steps=50,                    # in optimizer-step units
    evaluation_strategy="steps",
    eval_steps=500,                      # in optimizer-step units
    save_strategy="steps",
    save_steps=1000,                     # in optimizer-step units
    save_total_limit=50,
    # fp16=fp16,
    # bf16=bf16,
    gradient_checkpointing=True,         # saves VRAM (optional; slight compute overhead)
    dataloader_num_workers=4,
    report_to=["none"],                  # or "wandb", "tensorboard"
    ddp_find_unused_parameters=False,    # good default if you use DDP later
    seed=seed
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=probe_model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
)

# Helpful: set pad_token_id in model config to avoid warnings for CausalLM
if getattr(probe_model.config, "pad_token_id", None) is None:
    probe_model.config.pad_token_id = tokenizer.pad_token_id

trainer.train()

# (Optional) final save
trainer.save_model("out/final")
tokenizer.save_pretrained("out/final")

