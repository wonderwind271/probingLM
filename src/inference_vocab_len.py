from tokenizer.wordlevel_tokenizer import TrainableWordTokenizer
import json
import torch
import torch.nn.functional as F
from typing import List
import numpy as np
from transformers import (GPT2Config, GPT2LMHeadModel)
from model.probe_vocab import VocabProbingGPT2
from inference import word_list, add_tag, remove_last_occurrence
from matplotlib import pyplot as plt
from pathlib import Path


import matplotlib.pyplot as plt
from pathlib import Path

def plot_attention_weights(ls, title='Surprisal - Attached Natural Lens CE Loss'):
    plt.figure(figsize=(8, 5))
    colors = ['tab:blue', 'tab:orange']
    labels = ['simple context', 'normal context']

    for idx, (surprisal_list, label) in enumerate(zip(ls, labels)):
        x_coords = list(range(1, 13))  # 横坐标从 1 到 12
        plt.plot(x_coords, surprisal_list, label=label, color=colors[idx], marker='o')
        for i, value in enumerate(surprisal_list):
            plt.text(
                x_coords[i], value + 0.02,
                f'{value:.4f}',  # 保留四位小数
                fontsize=8,
                ha='center',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1.5)
            )

    plt.xticks(range(1, 13))
    plt.xlabel('Layer')
    plt.ylabel('Surprisal')
    plt.title(title)
    plt.legend()
    Path('figure').mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'figure/Attached Lens Surprisal CE loss 2.png')
    plt.clf()


@torch.no_grad()
def get_probe_surprisals(probe_model, tokenizer, context: str, target_token: str) -> List[float]:
    """
    Given a probe_model (VocabProbingGPT2), a context string, and a target token (must be a single token),
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
    assert len(all_probe_logits) == 1
    surprisals = []
    for logits in all_probe_logits:
        # Get logits for the last position (where next-token prediction happens)
        logits_next = logits[0, -1]  # shape: [vocab_size]
        log_probs = F.log_softmax(logits_next, dim=-1)
        surprisal = -log_probs[target_id].item()
        surprisals.append(surprisal)
        
    return surprisals


def probe_checkpoint_path_to_model(path, probing_layers):
    """Load probe checkpoint to model."""
    files = list(path.glob("checkpoint_3_*.pt"))
    assert len(files) == 1
    # if 10 in probing_layers:
    #     from IPython import embed;embed()
    checkpoint = torch.load(files[0], map_location=device)
    model = GPT2LMHeadModel(config=GPT2Config()).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    probe_model = VocabProbingGPT2(model, tokenizer, probing_layers=probing_layers, device=device)
    probe_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return probe_model


def cal_surprisal(tokenizer, simple=True):
    context_file_template = '/u501/x25luo/codebase/trabank-dev/test/word_context_archive/word_context{}.json'
    context_file_idxs = ['', '2', '5_0', '5_1', '5_2', '5_3', '5_4', '6_0', '6_1', '6_2']
    ls_surprisal = []
    surprisal_dict = {layer: {} for layer in range(12)}

    for layer in range(12):
        print(f'Start layer {layer} ......')
        probe_model = probe_checkpoint_path_to_model(Path(f'/u501/x25luo/codebase/probingLM/ckpt/vocab_len_childes_s42_layer{layer}_ce/'), probing_layers=[layer])
        probe_model.eval()
        layer_surprisal = []

        for file_idx in context_file_idxs:
            filename = context_file_template.format(file_idx)
            print('now process: '+filename)

            with open(filename) as fp:
                content = json.load(fp)
            updated_content = {}
            surprisal_dict[layer][f'context{file_idx}'] = dict()

            if simple:
                for k in content:
                    content[k]['env'] = k  # env have single word for childes, not vsdiag

            for word in word_list:
                env = content[word]['env'].replace('The child', '').replace('.', '')
                lan = content[word]['lan'].replace('"', '')
                lan = remove_last_occurrence(lan, word)
                updated_content[word] = {'env': env, 'lan': lan}

            for word, content in updated_content.items():
                context = '<CHI> '+add_tag(content['env'], ':<ENV>') + ' <CHI> ' + add_tag(content['lan'])
                target_token = add_tag(word)
                surprisals = get_probe_surprisals(probe_model, tokenizer, context, target_token)
                layer_surprisal += surprisals
                surprisal_dict[layer][f'context{file_idx}'][word] = surprisals[0]

        ls_surprisal.append(sum(layer_surprisal)/len(layer_surprisal))
    
    file_name = 'simple' if simple else 'normal'
    with open(f'/u501/x25luo/codebase/probingLM/surprisal_result/vocab_prob/{file_name}_context.json', 'w') as fp:
        json.dump(surprisal_dict, fp)

    print(ls_surprisal)
    return ls_surprisal


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    device = 'cpu'
    tokenizer = TrainableWordTokenizer(vocab_file='/u501/x25luo/codebase/probingLM/src/tokenizer/vocab.json')
    simple_sprl = cal_surprisal(tokenizer, simple=True)
    normal_sprl = cal_surprisal(tokenizer, simple=False)
    plot_attention_weights([simple_sprl, normal_sprl])

    # simple_gt = 4.304
    # normal_gt = 4.7339272850584235
    # plot_attention_weights([simple_sprl + [simple_gt], normal_sprl + [normal_gt]])

    # TODO: Welch's t-test