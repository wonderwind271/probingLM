'''Surprisal analysis of native lens. For every checkpoint of GPT2, we have 11 probes with size (d, V). We want to find the relationship between the performance of lens and that of the last layer, as well as the change with regard to training steps.'''
import os
from typing import Any, List
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from math import sqrt
from transformers import (AutoTokenizer, GPT2Config, GPT2LMHeadModel)
from model.probe_vocab import VocabProbingGPT2
from tokenizer.wordlevel_tokenizer import TrainableWordTokenizer
from inference_vocab_len import handle_template
from matplotlib import pyplot as plt
from inference import add_tag, remove_last_occurrence
import json


def plot(avg, stderr, step, folder_pth):
    x_coords = list(range(1, 13))
    plt.plot(x_coords, avg, marker='o', color='b')
    plt.fill_between(x_coords, avg-stderr, avg+stderr, color='b', alpha=0.15)
    plt.xlabel('Layer')
    plt.ylabel('Surprisal')
    plt.ylim(4, 12)
    plt.title(f'Surprisal of Different Layers at Step {step}')
    plt.legend()
    Path(folder_pth).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{folder_pth}/Native_Lens_Step{step}.png')
    print(f'saved {folder_pth}/Native_Lens_Step{step}.png')
    plt.clf()
    
    
def plot_all(avg_dict, folder_pth):
    x_coords = list(range(1, 13))
    for k, val in avg_dict.items():
        plt.plot(x_coords, val, label=f'Step {k}', marker='o')
        
    plt.xlabel('Layer')
    plt.ylabel('Surprisal')
    plt.ylim(4, 12)
    plt.title(f'Surprisal of Different Layers')
    plt.legend()
    Path(folder_pth).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{folder_pth}/Native_Lens_All_Steps.png')
    print(f'saved {folder_pth}/Native_Lens_All_Steps.png')
    plt.clf()


def create_subplots(avg_dict, folder_pth):
    fig, axs = plt.subplots(nrows=7, ncols=5, figsize=(21, 20))
    fig.suptitle('Surprisal of Different Layers at Different Steps', fontsize=20)
    
    x_coords = list(range(1, 13))

    for i, (k, val) in enumerate(avg_dict.items()):
        avg, stderr = val[0], val[1]
        row, col = i // 5, i % 5

        axs[row, col].plot(x_coords, avg, marker='o', color='b')
        axs[row, col].fill_between(x_coords, avg - stderr, avg + stderr, color='b', alpha=0.15)
        
        for x, y in zip(x_coords, avg):
            axs[row, col].text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
        
        axs[row, col].set_xlabel('Layer')
        axs[row, col].set_ylabel('Surprisal')
        axs[row, col].set_ylim(4, 12)
        axs[row, col].set_title(f'Step {k}')
        axs[row, col].legend()

    for i in range(len(avg_dict), len(axs.flatten())):
        row, col = i // 5, i % 5
        fig.delaxes(axs[row, col])
        
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整rect以给suptitle留出空间
    
    output_path = f'{folder_pth}/Combined_Surprisal_Plots.png'
    plt.savefig(output_path)
    print(f'saved {output_path}')
    plt.show()


def probe_checkpoint_path_to_model(path):
    """Load probe checkpoint to model."""
    checkpoint = torch.load(path, map_location=device)
    base_model = GPT2LMHeadModel(config=GPT2Config())
    base_model.resize_token_embeddings(len(tokenizer))
    probe_model = VocabProbingGPT2(base_model, tokenizer, probing_layers=checkpoint['probing_layers'], loss_type="ce",device=device, add_layernorm=True)
    probe_model.load_state_dict(checkpoint['model_state_dict'])
    probe_model.to(device)
    return probe_model


@torch.no_grad()
def get_probe_surprisals(probe_model, tokenizer, context: str, target_token: str) -> List[float]:
    """
    Given a probe_model (VocabProbingGPT2), a context string, and a target token (must be a single token),
    return a list of surprisals (negative log-probs) for that token at each probing layer.
    """
    # Tokenize context and target_token
    context_ids = tokenizer.encode(context, return_tensors='pt').to(probe_model.device)
    target_ids = tokenizer.encode(target_token, add_special_tokens=False)
    
    assert len(target_ids) == 1
    target_id = target_ids[0]
    # from IPython import embed;embed()
    base_output = probe_model.base_model(input_ids=context_ids, return_dict=True)
    gpt2_log_prob = F.log_softmax(base_output.logits[0, -1], dim=-1)
    gpt2_surprisal = -gpt2_log_prob[target_id].item()

    # Forward through probe model
    output = probe_model(input_ids=context_ids, labels=context_ids)
    all_probe_logits = output['all_probe_logits']

    surprisals = []
    for logits in all_probe_logits:
        # Get logits for the last position (where next-token prediction happens)
        logits_next = logits[0, -1]  # shape: [vocab_size]
        log_probs = F.log_softmax(logits_next, dim=-1)
        surprisal = -log_probs[target_id].item()
        surprisals.append(surprisal)
    
    return surprisals + [gpt2_surprisal]


def cal_surprisal(tokenizer, ckpt_path, simple=True):
    context_file_template = '/u501/x25luo/codebase/trabank-dev/test/word_context_archive/word_context{}.json'
    context_file_idxs = ['', '2', '5_0', '5_1', '5_2', '5_3', '5_4', '6_0', '6_1', '6_2']
    surprisal_dict = {}

    probe_model = probe_checkpoint_path_to_model(ckpt_path)
    probe_model.eval()

    for file_idx in context_file_idxs:
        filename = context_file_template.format(file_idx)
        print('now process: '+filename)
        updated_content = handle_template(filename, simple)

        for word, content in updated_content.items():
            context = '<CHI> '+add_tag(content['env'], ':<ENV>') + ' <CHI> ' + add_tag(content['lan'])
            target_token = add_tag(word)
            surprisals = get_probe_surprisals(probe_model, tokenizer, context, target_token)
            surprisal_dict[f'f{file_idx}_{word}'] = surprisals
            
    file_name = 'simple' if simple else 'normal'
    with open(f'/u501/x25luo/codebase/probingLM/surprisal_result/native_lens/{file_name}_context_step{step}.json', 'w') as fp:
        json.dump(surprisal_dict, fp)

    return surprisal_dict


def surprisal_stat(ls:list):
    sprl_arr = np.array(ls)
    N, D = sprl_arr.shape
    assert D == 12
    return np.mean(sprl_arr, axis=0), np.std(sprl_arr, axis=0)/sqrt(N)


def plot_20k_30k(folder_pth):
    '''only plot lines under 20k and 30k steps in a picture'''
    plot_ls= []
    for step in [20000, 30000]:
        with open(f'/u501/x25luo/codebase/probingLM/surprisal_result/native_lens/normal_context_step{step}.json', 'r') as fp:
            surprisal_dict = json.load(fp)
        surprisal_ls = list(surprisal_dict.values())
        avg, stderr = surprisal_stat(surprisal_ls)
        plot_ls.append((avg, stderr, step))
    
    x_coords = list(range(1, 13))
    for (avg, stderr, step) in plot_ls:
        plt.plot(x_coords, avg, marker='o', label=f'Step {step}')
        plt.fill_between(x_coords, avg-stderr, avg+stderr, alpha=0.15)
    plt.grid()
    plt.xlabel('Layer')
    plt.ylabel('Surprisal')
    plt.title(f'Surprisal of Different Layers at Step 20k and 30k')
    plt.legend()
    Path(folder_pth).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{folder_pth}/Native Lens Step 20k vs 30k.png')
    plt.clf()
        
        
if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('device:', device)
    # tokenizer = TrainableWordTokenizer(vocab_file='/u501/x25luo/codebase/probingLM/src/tokenizer/vocab.json')
    # CHECKPOINT_DIR = '/u501/x25luo/codebase/probingLM/ckpt/native_lens/output/seed42'
    
    # pt_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
    # ckpt_ls = sorted(pt_files, key=lambda x: int(x.split('_')[-1].replace('.pt', '')))

    # all_step_avg = {}
    
    # for pth in ckpt_ls:
    #     print('*** Currently dealing with checkpoint', pth)
    #     if str(pth).endswith('20000.pt'):
    #         step = int(pth.split('_')[-1].replace('.pt', ''))
    #         checkpoint_path = os.path.join(CHECKPOINT_DIR, pth)
    #         surprisal_dict = cal_surprisal(tokenizer, checkpoint_path, simple=False)
    #         surprisal_ls = list(surprisal_dict.values())
    #         avg, stderr = surprisal_stat(surprisal_ls)
    #         plot(avg, stderr, step, 'figure/native_len_CHILDS_bf16/')
    #         all_step_avg[step] = (avg, stderr)
        
    # create_subplots(all_step_avg, 'figure/native_len_CHILDS_bf16/')

    plot_20k_30k('figure/native_len_CHILDS_bf16/')        
