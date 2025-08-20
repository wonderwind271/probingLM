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
from inference import add_tag
import json
from compare import all_probing_s42, all_probing_s142, all_probing_s242


def plot(avg, stderr, step, folder_pth):
    x_coords = list(range(1, 13))
    plt.plot(x_coords, avg, marker='o', color='b')
    plt.fill_between(x_coords, avg-stderr, avg+stderr, color='b', alpha=0.15)
    for _x, _y in zip(x_coords, avg):
        plt.text(_x, _y, f'{_y:.2f}', ha='center', va='bottom', fontsize=6,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.2))
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


def create_subplots(avg_dict, folder_pth):
    fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(20,20))
    fig.suptitle('Surprisal of Different Layers at Different Steps', fontsize=20)
    
    x_coords = list(range(1, 13))

    for i, (k, val) in enumerate(avg_dict.items()):
        avg, stderr = val[0], val[1]
        row, col = i // 4, i % 4
        
        axs[row, col].plot(x_coords, avg, marker='o', color='b')
        axs[row, col].fill_between(x_coords, avg - stderr, avg + stderr, color='b', alpha=0.15)
        
        for x, y in zip(x_coords, avg):
            axs[row, col].text(x, y+0.1, f'{y:.2f}', ha='center', va='bottom', fontsize=7,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.2))
        
        axs[row, col].set_xlabel('Layer')
        axs[row, col].set_ylabel('Surprisal')
        axs[row, col].set_ylim(4, 12)
        axs[row, col].set_title(f'Step {k}')
        axs[row, col].legend()

    # delete unused subfigures
    # for i in range(len(avg_dict), len(axs.flatten())):
    #     row, col = i // 4, i % 5
    #     fig.delaxes(axs[row, col])
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
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
    from IPython import embed;embed()
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
            from IPython import embed;embed()
            context = '<CHI> '+add_tag(content['env'], ':<ENV>') + ' <CHI> ' + add_tag(content['lan'])
            target_token = add_tag(word)
            surprisals = get_probe_surprisals(probe_model, tokenizer, context, target_token)
            surprisal_dict[f'f{file_idx}_{word}'] = surprisals
            
    file_name = 'simple' if simple else 'normal'
    with open(f'/u501/x25luo/codebase/probingLM/surprisal_result/native_lens/seed242/{file_name}_context_step{step}.json', 'w') as fp:
        json.dump(surprisal_dict, fp)

    return surprisal_dict


def surprisal_stat(ls:list):
    sprl_arr = np.array(ls)
    N, D = sprl_arr.shape
    assert D == 12
    return np.mean(sprl_arr, axis=0), np.std(sprl_arr, axis=0)/sqrt(N)


def compare_native_tuned_lens(folder_pth = 'figure'):
    '''Compare native lens with tuned lens under 0, 5k, 10k, 15k, 20k steps'''
    avg_tuned_lens = np.array([all_probing_s42, all_probing_s142, all_probing_s242])
    avg_tuned_lens = np.mean(avg_tuned_lens, axis=0)
    for seed in [42, 142, 242]:
        with open(f'/u501/x25luo/codebase/probingLM/surprisal_result/native_lens/seed{seed}/normal_context_step20000.json', 'r') as fp:
            surprisal_dict = json.load(fp)
        surprisal_ls = list(surprisal_dict.values())
            
    plot_ls= {}
    for idx, step in enumerate([1000, 5000, 10000, 15000, 20000]):
        avg_native_lens_ls = []
        for seed in [42, 142, 242]:
            with open(f'/u501/x25luo/codebase/probingLM/surprisal_result/native_lens/seed{seed}/normal_context_step{step}.json', 'r') as fp:
                surprisal_dict = json.load(fp)
            surprisal_ls = list(surprisal_dict.values())
            avg_native_lens, _ = surprisal_stat(surprisal_ls)
            avg_native_lens_ls.append((avg_native_lens))
        avg_native_lens = np.array(avg_native_lens_ls).mean(axis = 0)
        plot_ls[step] = (avg_native_lens, avg_tuned_lens[idx])
        
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12,12))
    fig.suptitle('Surprisal of Different Layers at Different Steps', fontsize=16)
    
    x_coords = list(range(1, 13))

    for i, (k, val) in enumerate(plot_ls.items()):
        native_lens, tuned_lens = val[0], val[1]
        row, col = i // 2, i % 2
        
        axs[row, col].plot(x_coords, native_lens, marker='o', label = 'Native Lens')
        axs[row, col].plot(x_coords, tuned_lens, marker='o', label = 'Tuned Lens')
        
        for x, y in zip(x_coords, native_lens):
            axs[row, col].text(x, y+0.1, f'{y:.2f}', ha='center', va='bottom', fontsize=7,
                               bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.2))
        for x, y in zip(x_coords, tuned_lens):
            axs[row, col].text(x, y-0.1, f'{y:.2f}', ha='center', va='top', fontsize=7,
                               bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.2))
        
        axs[row, col].set_xlabel('Layer')
        axs[row, col].set_ylabel('Surprisal')
        axs[row, col].set_ylim(4, 12)
        axs[row, col].set_title(f'Step {k}')
        axs[row, col].legend()
            
    fig.delaxes(axs[2, 1])
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = f'{folder_pth}/Compare_Native_Tuned_Lens.png'
    plt.savefig(output_path)
    print(f'saved {output_path}')
    plt.clf()
    
    
def plot_single_step2k(folder_pth = 'figure'):
    '''Compare 3 settings under step 20k'''
    x_coords = list(range(1, 13))
    
    with open(f'/u501/x25luo/codebase/probingLM/surprisal_result/native_lens/seed42/normal_context_step20000.json', 'r') as fp:
        surprisal_dict = json.load(fp)
    surprisal_ls = list(surprisal_dict.values())
    native_lens, native_lens_stderr = surprisal_stat(surprisal_ls)
    
    # d*d tuned lens
    tuned_lens = np.array(all_probing_s42[4])
    
    plt.figure(figsize=(10,6))
    plt.plot(x_coords, native_lens, marker='o', label = 'Native Lens', color='r')
    plt.fill_between(x_coords, native_lens+native_lens_stderr,
                     native_lens-native_lens_stderr, color='r', alpha=0.15)
    
    plt.plot(x_coords, tuned_lens, marker='o', label = 'Tuned Lens, d*d', color='b')
    
    for x, y in zip(x_coords, native_lens):
        plt.text(x, y+0.1, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='r',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.2))
    # for x, y in zip(x_coords, tuned_lens):
    #     plt.text(x, y-0.1, f'{y:.2f}', ha='center', va='top', fontsize=7, color='b',
    #             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.2))
    
    # d*V tuned lens
    from t_test import read_surprisal
    stat_ls, _ = read_surprisal()
    
    stat = np.array(stat_ls)
    stderr = stat[:, 1]/np.sqrt(stat[:, 2])
    plt.plot(x_coords, stat[:, 0], marker='o', label = 'Tuned Lens, d*V', color='g')
    plt.fill_between(x_coords, stat[:, 0]+stderr, stat[:, 0]-stderr, color='g', alpha=0.15)

    plt.xlabel('Layer')
    plt.ylabel('Surprisal')
    plt.title(f'Surprisal of Different Layers at Step 20000')
    plt.legend()
    
    plt.tight_layout()
    output_path = f'{folder_pth}/Compare_3_Settings_at_20k.png'
    plt.savefig(output_path)
    plt.clf()


if __name__ == '__main__':
    # ------ plottig ------
    compare_native_tuned_lens()
    plot_single_step2k(folder_pth = 'figure')
    
    # ------ test surprisal ------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    tokenizer = TrainableWordTokenizer(vocab_file='/u501/x25luo/codebase/probingLM/src/tokenizer/vocab.json')
    CHECKPOINT_DIR = '/u501/x25luo/codebase/probingLM/ckpt/native_lens/output/seed242'
    
    pt_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
    ckpt_ls = sorted(pt_files, key=lambda x: int(x.split('_')[-1].replace('.pt', '')))

    all_step_avg = {}
    
    for pth in ckpt_ls:
        print('*** Currently dealing with checkpoint', pth)
            
        step = int(pth.split('_')[-1].replace('.pt', ''))
        checkpoint_path = os.path.join(CHECKPOINT_DIR, pth)
        # try:
        #     with open(f'/u501/x25luo/codebase/probingLM/surprisal_result/native_lens/seed242/normal_context_step{step}.json', 'r') as fp:
        #         surprisal_dict = json.load(fp)
        # except:
        surprisal_dict = cal_surprisal(tokenizer, checkpoint_path, simple=False)
        surprisal_ls = list(surprisal_dict.values())
        avg, stderr = surprisal_stat(surprisal_ls)
        plot(avg, stderr, step, 'figure/native_len_CHILDS_seed242/')
        all_step_avg[step] = (avg, stderr)
        
    create_subplots(all_step_avg, 'figure/native_len_CHILDS_seed242/')