# from trainer import *
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
                          get_scheduler, AutoModelForCausalLM)

import torch.nn.functional as F
from typing import List
from model.probe import ProbingOutput, LensProbingGPT2, NaturalProbingGPT2
from tokenizer.wordlevel_tokenizer import TrainableWordTokenizer
import json
tokenizer = AutoTokenizer.from_pretrained('gpt2')
device = torch.device('cuda')
def probe_checkpoint_path_to_model(path, probing_layers, mode='pt'):
    """Load probe checkpoint to model."""
    checkpoint = torch.load(path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2LMHeadModel(config=GPT2Config(n_layer=12)).to(device)
    # model.resize_token_embeddings(len(tokenizer))
    model.eval()
    probe_model = LensProbingGPT2(model, tokenizer, probing_layers=probing_layers, device=device).to(device)
    if mode == 'pt':
        probe_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        probe_model.load_state_dict(checkpoint, strict=False)
    return probe_model

oid = 0
# layer = [5,10,16][oid]
cid = 12
seed = 442

for oid in range(4):
    item_name = f'natural_wiki_tunedlens_{oid}'
    # options = [[0,1,2,3,4,5], [6,7,8,9,10],[11,12,13,14,15,16]]
    probing_layer = [list(range(-1,2)), list(range(2,5)), list(range(5,8)), list(range(8,11))]
    probing_layer = probing_layer[oid]
    probe_model = probe_checkpoint_path_to_model(f'/scratch/chaijy_root/chaijy2/shuyuwu/experiments/checkpoints/{item_name}/checkpoint-100000/pytorch_model.bin', probing_layers=probing_layer, mode='bin')
    probe_model.eval()


    @torch.no_grad()
    def get_probe_surprisals(probe_model, tokenizer, context: str, target_token: str) -> List[float]:
        """
        Given a probe_model (LensProbingGPT2), a context string, and a target token (must be a single token),
        return a list of surprisals (negative log-probs) for that token at each probing layer.
        """

        # Tokenize context and target_token
        context_ids = tokenizer.encode(context, return_tensors='pt').to(probe_model.device)
        target_ids = tokenizer.encode(target_token, add_special_tokens=False)
        
        if len(target_ids) != 1:
            raise ValueError("target_token must be a single token under this tokenizer")

        target_id = target_ids[0]


        # Forward through probe model
        output = probe_model(input_ids=context_ids)
        all_probe_logits = output.all_probe_logits  # List[Tensor], each [1, seq_len, vocab_size]

        surprisals = []
        for logits in all_probe_logits:
            # Get logits for the last position (where next-token prediction happens)
            logits_next = logits[0, -1]  # shape: [vocab_size]
            log_probs = F.log_softmax(logits_next, dim=-1)
            surprisal = -log_probs[target_id].item()
            surprisals.append(surprisal)

        return surprisals

    def remove_last_occurrence(lan: str, word: str) -> str:
        """Remove last occurrence of word in lan."""
        lower_lan = lan.lower()
        lower_word = word.lower()

        pos = lower_lan.rfind(lower_word)

        # If the word is found, return everything before that occurrence.
        # Otherwise, return the original string.
        return lan[:pos] if pos != -1 else lan

    def add_tag(text, tag=':<LAN>'):
        """Add tag to plain text."""
        words = text.split()
        for i in range(len(words)):
            words[i] += tag
        return ' '.join(words)

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

    local_dataset_name = "nyu-mll/blimp"
    for text_mode in ['bad', 'good']:
    # text_mode = 'bad'
        subset_name = 'animate_subject_passive'
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
        np.save(f"results_wiki_tunedlens_{oid}_blimp_{subset_name}_{text_mode}.npy", arr)   # binary, efficient
