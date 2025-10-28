import json
import os
import matplotlib.pyplot as plt
import numpy as np

word_list2 = ['box', 'book', 'ball', 'hand', 'paper', 'table', 'toy', 'head', 'car', 'chair', 'room', 'picture', 'doll', 'cup', 'towel', 'door', 'mouth', 'camera', 'duck', 'face', 'truck', 'bottle', 'puzzle', 'bird', 'tape', 'finger', 'bucket', 'block', 'stick', 'elephant', 'hat', 'bed', 'arm', 'dog', 'kitchen', 'spoon', 'hair', 'blanket', 'horse', 'tray', 'train', 'cow', 'foot', 'couch', 'necklace', 'cookie', 'plate', 'telephone', 'window', 'brush', 'ear', 'pig', 'purse', 'hammer', 'cat', 'shoulder', 'garage', 'button', 'monkey', 'pencil', 'shoe', 'drawer', 'leg', 'bear', 'milk', 'egg', 'bowl', 'juice', 'ladder', 'basket', 'coffee', 'bus', 'food', 'apple', 'bench', 'sheep', 'airplane', 'comb', 'bread', 'eye', 'animal', 'knee', 'shirt', 'cracker', 'glass', 'light', 'game', 'cheese', 'sofa', 'giraffe', 'turtle', 'stove', 'clock', 'star', 'refrigerator', 'banana', 'napkin', 'bunny', 'farm', 'money']  # 100 in total. from childes_word_list intersect vsdiag vocab intersect CDI nouns catagory and take first 100
context_file_idxs = ['', '2', '5_0', '5_1', '5_2', '5_3', '5_4', '6_0', '6_1', '6_2']
result_template = 'context{}_list2_envsingle_result.json'
seed_list = list(range(42,443,100))
cid_list = [12,22,32,42]
layer_surprisals = [[] for _ in range(11)]

for seed in seed_list:
    
    if seed !=42:
        dir_template = 'probe_result/childes_warmup_s{seed}_c{cid}_kl_shuffled_tunedlens_layer{layer_num}'
    else:
        dir_template = 'probe_result/childes_warmup_s{seed}_c{cid}_shuffled_tunedlens_layer{layer_num}'

    cid = 12
    analysis_dirs = [(dir_template.format(seed=seed, cid=cid, layer_num=5), 6), (dir_template.format(seed=seed, cid=cid, layer_num=10), 5), (dir_template.format(seed=seed, cid=cid, layer_num=16), 6)]
    analysis_dir2 = [(dir_template.format(seed=seed, cid=cid, layer_num=5), 6), (dir_template.format(seed=seed, cid=cid, layer_num=10), 5)]
    all_res = []

    for dir, layer_num in analysis_dir2:
        all_res_dir = [[] for i in range(layer_num)]
        for file_idx in context_file_idxs:
            filename = dir+'/'+result_template.format(file_idx)
            with open(filename) as fp:
                file_res = json.load(fp)
                for word in file_res:
                    values = file_res[word]
                    for i, surprisal in enumerate(values):
                        all_res_dir[i].append(surprisal)
        all_res += all_res_dir
    for i in range(11):
        layer_surprisals[i] += all_res[i]
    
# print(all_res)
# actual_surprisal = 6.384129475414753
# plt.xticks(list(range(1,19)))  
# plt.plot(list(range(1,19)), all_res+[actual_surprisal])
# plt.xlabel('layer')
# plt.ylabel('avg surprisal')
# plt.title(f'layer probing (KL) on step 20000 for seed {seed}')
# plt.show()
print(len(layer_surprisals[0]))
stds = []
for i in range(11):
    stds.append(np.std(layer_surprisals[i], ddof=1)/np.sqrt(5000))
print(stds)

