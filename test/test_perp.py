from model.probe import ProbingOutput, NaturalProbingGPT2

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
import numpy as np

# torch.set_default_device('cuda')
def probe_checkpoint_path_to_model(path, probing_layers, tokenizer):
    """Load probe checkpoint to model."""
    checkpoint = torch.load(path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2LMHeadModel(config=GPT2Config(n_layer=12)).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    probe_model = NaturalProbingGPT2(model, tokenizer, probing_layers=probing_layers, device=device).to(device)
    probe_model.load_state_dict(checkpoint)
    probe_model.to(device)
    return probe_model



base_model = GPT2LMHeadModel(config=GPT2Config())
# tokenizer = AutoModelForCausalLM.from_pretrained(model_name)
# wrap up `probe model`
probing_layers = list(range(-1,11))

# for cid in range(1000,100001,1000):

cid = 100000

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ckpt_dir = f"/scratch/chaijy_root/chaijy2/shuyuwu/experiments/checkpoints/natural_wiki/checkpoint-{cid}/pytorch_model.bin"
tokenizer = AutoTokenizer.from_pretrained('gpt2')
probe_model = probe_checkpoint_path_to_model(ckpt_dir, probing_layers, tokenizer).to(device)
real_base = probe_model.base_model
# checkpoint_path = f'/scratch/chaijy_root/chaijy2/shuyuwu/experiments/checkpoints/natural_wiki/base/checkpoint-{cid}.pt'
# model_state_dict = real_base.module.state_dict() if hasattr(real_base, 'module') else real_base.state_dict()
# torch.save(model_state_dict, checkpoint_path)
# print(f"checkpoint {cid} saved to {checkpoint_path}")

# exit()

seed = 42

item_name = f'childes_s{seed}_c{cid}_kl_nativelens'

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

def get_perp(model, tokenizer, sentence, max_length=1024):
    # Resolve device from the model
    # device = next(model.parameters()).device

    # Tokenize
    enc = tokenizer(
        sentence,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        padding=False,  # single sentence -> no padding
    )
    # labels = input_ids (causal LM); if you add padding later, also mask -100 where padding==0
    enc["labels"] = enc["input_ids"].clone()

    # Move to device
    enc = {k: v.to(device) for k, v in enc.items()}
    # print(enc)
    # print(enc['input_ids'].device)
    # print(next(model.parameters()).device)
    # print(model.base_model.device)
    with torch.no_grad():
        out = model(**enc)  # returns your ProbingOutput

    loss_list = []
    labels = enc["labels"]  # use labels (not raw input_ids)


    # Per-probe (each logits is BxTxV)
    for logits in out.all_probe_logits:
        shifted_logits = logits[:, :-1, :]     # (B, T-1, V)
        shifted_labels = labels[:, 1:]         # (B, T-1)

        loss_i = loss_fn(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            shifted_labels.reshape(-1)
        )
        loss_list.append(float(loss_i.item()))

    # Final layer loss (provided by your model)
    if out.loss_main is not None:
        final_loss = out.loss_main.detach().float().item()
        loss_list.append(final_loss)


    return loss_list

# local_dataset_name = "wonderwind271/wikipedia-selection-512w"
# raw = load_dataset(local_dataset_name)['test']

# results = []

# for idx, record in enumerate(raw):
#     text = record['text']
#     loss_list = get_perp(probe_model, tokenizer, text)
#     # print(loss_list)
#     results.append(loss_list)
#     if idx % 100 == 0:
#         print(f'{idx} completed')


# arr = np.array(results)       # shape (N, M)
# np.save("results_100000.npy", arr)   # binary, efficient
local_dataset_name = "nyu-mll/blimp"
for text_mode in ['bad', 'good']:
# text_mode = 'bad'
    subset_name = 'anaphor_number_agreement'
    raw = load_dataset(local_dataset_name, subset_name)['train']

    results = []

    for idx, record in enumerate(raw):
        text = record[f'sentence_{text_mode}']
        loss_list = get_perp(probe_model, tokenizer, text)
        # print(loss_list)
        results.append(loss_list)
        if idx % 100 == 0:
            print(f'{idx} completed')


    arr = np.array(results)       # shape (N, M)
    np.save(f"results_wiki_nativelens_blimp_{subset_name}_{text_mode}.npy", arr)   # binary, efficient