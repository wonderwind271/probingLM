from tokenizer.wordlevel_tokenizer import TrainableWordTokenizer
import json
import torch
import torch.nn.functional as F
from typing import List
import numpy as np
from transformers import (GPT2Config, GPT2LMHeadModel)
from model.probe import LensProbingGPT2
import os


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


def probe_checkpoint_path_to_model(path, probing_layers):
    """Load probe checkpoint to model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device)
    model = GPT2LMHeadModel(config=GPT2Config()).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    probe_model = LensProbingGPT2(model, tokenizer, probing_layers=probing_layers).to(device)
    probe_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return probe_model


word_list = ['box', 'book', 'ball', 'hand', 'paper', 'table', 'toy', 'head', 'car', 'chair', 'room', 'picture', 'doll', 'cup', 'towel', 'door', 'mouth', 'camera', 'duck', 'face', 'truck', 'bottle', 'puzzle', 'bird', 'tape', 'finger', 'bucket', 'block', 'stick', 'elephant', 'hat', 'bed', 'arm', 'dog', 'kitchen', 'spoon', 'hair', 'blanket', 'horse', 'tray', 'train', 'cow', 'foot', 'couch', 'necklace', 'cookie', 'plate', 'telephone', 'window', 'brush', 'ear', 'pig', 'purse', 'hammer', 'cat', 'shoulder', 'garage', 'button', 'monkey', 'pencil', 'shoe', 'drawer', 'leg', 'bear', 'milk', 'egg', 'bowl', 'juice', 'ladder', 'basket', 'coffee', 'bus', 'food', 'apple', 'bench', 'sheep', 'airplane', 'comb', 'bread', 'eye', 'animal', 'knee', 'shirt', 'cracker', 'glass', 'light', 'game', 'cheese', 'sofa', 'giraffe', 'turtle', 'stove', 'clock', 'star', 'refrigerator', 'banana', 'napkin', 'bunny', 'farm', 'money']  # 100 in total. from childes_word_list intersect vsdiag vocab intersect CDI nouns catagory and take first 100


def overall_surprisal():
    context_file_template = '/u501/x25luo/codebase/probingLM/result_s42_6_10/context{}_list2_envsingle_result.json'
    context_file_idxs = ['', '2', '5_0', '5_1', '5_2', '5_3', '5_4', '6_0', '6_1', '6_2']
    avg_surprisal = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    overall_item = 0

    for file_idx in context_file_idxs:
        result_json = context_file_template.format(file_idx)
        with open(result_json, 'r') as fp:
            surprisal = json.load(fp)
        overall_item += len(surprisal)
        for k in surprisal:
            avg_surprisal += np.array(surprisal[k])
    print('total item:', overall_item)
    print('surprisal in step 6-10')
    print(avg_surprisal/overall_item)


if __name__ == '__main__':
    tokenizer = TrainableWordTokenizer(vocab_file='/u501/x25luo/codebase/probingLM/src/tokenizer/vocab.json')
    probe_model = probe_checkpoint_path_to_model('/u501/x25luo/codebase/probingLM/ckpt/childes_warmup_s42_shuffled/checkpoint_0_2836.pt', probing_layers=[6,7,8,9,10])
    print('### checkpoint_1_5672.pt')
    probe_model.eval()

    context_file_template = '/u501/x25luo/codebase/trabank-dev/test/word_context_archive/word_context{}.json'
    context_file_idxs = ['', '2', '5_0', '5_1', '5_2', '5_3', '5_4', '6_0', '6_1', '6_2']
    
    dir_path = f'/u501/x25luo/codebase/probingLM/result_s42_6_10'
    os.makedirs(dir_path, exist_ok=True)
    result_template = dir_path+'/context{}_list2_envsingle_result.json'
    updated_context_file_template = 'other/word_context{}_updated.json'

    for file_idx in context_file_idxs:
        filename = context_file_template.format(file_idx)
        result_json = result_template.format(file_idx)
        print('now process: '+filename)
        result_dict = {}
        with open(filename) as fp:
            content = json.load(fp)
        all_env = []
        all_lan = []
        updated_content = {}

        for k in content:
            content[k]['env'] = k  # env have single word for childes, not vsdiag

        for word in word_list:
            env = content[word]['env'].replace('The child', '').replace('.', '')
            lan = content[word]['lan'].replace('"', '')
            lan = remove_last_occurrence(lan, word)
            updated_content[word] = {'env': env, 'lan': lan}
            all_env.append(env)
            all_lan.append(lan)

        # with open(updated_context_file_template.format(file_idx), 'w') as fp:
        #     json.dump(updated_content, fp)

        surprisal_list = []
        env_list = all_env
        lan_list = all_lan
        for word_idx, cur_env in enumerate(env_list):
            cur_lan = lan_list[word_idx]
            context = '<CHI> '+add_tag(cur_env, ':<ENV>') + ' <CHI> ' + add_tag(cur_lan)
            cur_word = word_list[word_idx]
            target_token = add_tag(cur_word)
            surprisals = get_probe_surprisals(probe_model, tokenizer, context, target_token)
            result_dict[cur_word] = surprisals
            if (word_idx+1) % 10 == 0:
                print(f'completed {word_idx+1} / 100 for {filename}')
        with open(result_json, 'w') as fp:
            json.dump(result_dict, fp)

    overall_surprisal()