# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun --standalone --nproc_per_node=4 train_hf_trainer.py


from model.probe import ProbingOutput, NaturalProbingGPT2, LensProbingGPT2

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

# torch.set_default_device('cuda')


def checkpoint_path_to_model(path):
    """Load checkpoint to model."""
    checkpoint = torch.load(path)
    model = GPT2LMHeadModel(config=GPT2Config(n_layer=12))
    # model.resize_token_embeddings(len(tokenizer))
    # model.to(device)
    model.load_state_dict(checkpoint)
    return model

# -----------------------------
# Model & Tokenizer
# -----------------------------
# If you already have `model` instantiated, comment out the next line and import/construct yours.
model_name = "gpt2"
model_dir = f'/scratch/chaijy_root/chaijy2/shuyuwu/experiments/checkpoints/natural_wiki/base/checkpoint-100000.pt'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

oid = 0

base_model = checkpoint_path_to_model(model_dir).to(device)
tokenizer = AutoModelForCausalLM.from_pretrained(model_name)
# wrap up `probe model`
probing_layer = [list(range(-1,2)), list(range(2,5)), list(range(5,8)), list(range(8,11))]
probing_layer = probing_layer[oid]

probe_model = LensProbingGPT2(base_model, tokenizer, probing_layers=probing_layer, device=device).to(device)

# If you already have a tokenizer, comment this and use yours.
tokenizer = AutoTokenizer.from_pretrained(model_name)
# GPT-2 has no pad token; map pad -> eos for LM training.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Data
# -----------------------------
# Small, quick dataset for demo. Replace with your dataset when ready.
local_dataset_name = "wonderwind271/wikipedia-selection-512w"
# raw = load_dataset(local_dataset_name)['train']
# Train/val splits are "train" and "validation" here.

# Simple tokenizer function
def tok_fn(examples):
    # Truncate to a reasonable length; adjust as needed
    return tokenizer(examples["text"], truncation=True, max_length=1024)

# tokenized = raw.map(tok_fn, batched=True, remove_columns=["text"])

tokenized = load_from_disk('/scratch/chaijy_root/chaijy2/shuyuwu/temp_map_wiki')

# tokenized.save_to_disk('/scratch/chaijy_root/chaijy2/shuyuwu/temp_map_wiki')

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -----------------------------
# Hyperparameters
# -----------------------------
per_device_bs = 16                 # what fits in VRAM
grad_accum = 1                    # 8 * 2 = 16 effective per GPU
max_steps = 100000                 # keep steps the same; accumulation doesn't change total optimizer steps
warmup_ratio = 0.05               # easier than counting steps manually
learning_rate = 5e-5              # adjust to your model/task
weight_decay = 0.05
max_grad_norm = 2.5
seed = 42

# Mixed precision—set bf16=True on newer GPUs (A100/H100), otherwise fp16=True
# use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+
# fp16 = not use_bf16
# bf16 = use_bf16

# -----------------------------
# TrainingArguments
# -----------------------------
args = TrainingArguments(
    output_dir=f"/scratch/chaijy_root/chaijy2/shuyuwu/experiments/checkpoints/natural_wiki_tunedlens_{oid}",
    overwrite_output_dir=True,
    per_device_train_batch_size=per_device_bs,
    # per_device_eval_batch_size=per_device_bs,
    gradient_accumulation_steps=grad_accum,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,                 # use max_steps to match your planned 20k optimizer updates
    warmup_ratio=warmup_ratio,           # counted in optimizer-step units
    lr_scheduler_type="linear",
    logging_steps=10,                    # in optimizer-step units
    # evaluation_strategy="steps",
    # eval_steps=500,                      # in optimizer-step units
    save_strategy="steps",
    save_steps=5000,                     # in optimizer-step units
    save_total_limit=500,
    # fp16=fp16,
    # bf16=bf16,
    # gradient_checkpointing=True,         # saves VRAM (optional; slight compute overhead)
    dataloader_num_workers=4,
    report_to=["wandb"],                  # or "wandb", "tensorboard"
    ddp_find_unused_parameters=False,    # good default if you use DDP later
    seed=seed,
    run_name=f'natural-wiki-42-tunedlens-{oid}',
    evaluation_strategy="no",
    save_safetensors=False,
)

# -----------------------------
# Trainer
# -----------------------------

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        val = getattr(outputs, "loss_main", None)
        if val is not None:
            if hasattr(val, "detach"):
                val = val.detach()
            try:
                # if it's a tensor: make scalar; if it's already a number, float() works
                v = val.item() if hasattr(val, "numel") and val.numel() == 1 else float(val.mean().item())
            except Exception:
                # final fallback: try float() (handles plain numbers)
                v = float(val)
            self.log({"main_loss": v})

        return (loss, outputs) if return_outputs else loss

trainer = MyTrainer(
    model=probe_model,
    args=args,
    train_dataset=tokenized,
    # eval_dataset=tokenized["test"],
    data_collator=data_collator,

)

# Helpful: set pad_token_id in model config to avoid warnings for CausalLM
if getattr(probe_model.base_model.config, "pad_token_id", None) is None:
    probe_model.base_model.config.pad_token_id = tokenizer.pad_token_id

trainer.train()

# # (Optional) final save
# trainer.save_model("/scratch/chaijy_root/chaijy2/shuyuwu/experiments/checkpoints/natural_wiki/final")
# tokenizer.save_pretrained("/scratch/chaijy_root/chaijy2/shuyuwu/experiments/checkpoints/natural_wiki/final")


