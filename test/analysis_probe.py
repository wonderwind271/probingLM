import json
import os
import matplotlib.pyplot as plt

word_list2 = ['box', 'book', 'ball', 'hand', 'paper', 'table', 'toy', 'head', 'car', 'chair', 'room', 'picture', 'doll', 'cup', 'towel', 'door', 'mouth', 'camera', 'duck', 'face', 'truck', 'bottle', 'puzzle', 'bird', 'tape', 'finger', 'bucket', 'block', 'stick', 'elephant', 'hat', 'bed', 'arm', 'dog', 'kitchen', 'spoon', 'hair', 'blanket', 'horse', 'tray', 'train', 'cow', 'foot', 'couch', 'necklace', 'cookie', 'plate', 'telephone', 'window', 'brush', 'ear', 'pig', 'purse', 'hammer', 'cat', 'shoulder', 'garage', 'button', 'monkey', 'pencil', 'shoe', 'drawer', 'leg', 'bear', 'milk', 'egg', 'bowl', 'juice', 'ladder', 'basket', 'coffee', 'bus', 'food', 'apple', 'bench', 'sheep', 'airplane', 'comb', 'bread', 'eye', 'animal', 'knee', 'shirt', 'cracker', 'glass', 'light', 'game', 'cheese', 'sofa', 'giraffe', 'turtle', 'stove', 'clock', 'star', 'refrigerator', 'banana', 'napkin', 'bunny', 'farm', 'money']  # 100 in total. from childes_word_list intersect vsdiag vocab intersect CDI nouns catagory and take first 100
context_file_idxs = ['', '2', '5_0', '5_1', '5_2', '5_3', '5_4', '6_0', '6_1', '6_2']
result_template = 'context{}_list2_envsingle_result.json'

analysis_dirs = [('probe_s42_0_5', 6), ('probe_s42_6_10', 5)]
all_res = []

for dir, layer_num in analysis_dirs:
    all_res_dir = [0 for i in range(layer_num)]
    for file_idx in context_file_idxs:
        filename = dir+'/'+result_template.format(file_idx)
        with open(filename) as fp:
            file_res = json.load(fp)
            for word in file_res:
                values = file_res[word]
                for i, surprisal in enumerate(values):
                    all_res_dir[i] += surprisal
    for i in range(layer_num):
        all_res_dir[i] /= 1000
    all_res += all_res_dir
    
print(all_res)
plt.plot(list(range(1,13)), all_res+[4.3])
plt.xlabel('layer')
plt.ylabel('avg surprisal')
plt.show()
