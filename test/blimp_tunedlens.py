import numpy as np
import matplotlib.pyplot as plt

all_layers_score = []
subject = 'animate_subject_passive'
all_score = []

for i in range(4):
    all_perp_bad = np.load(f'result_blimp_tunedlens/results_wiki_tunedlens_{i}_blimp_{subject}_bad.npy')
    all_perp_good = np.load(f'result_blimp_tunedlens/results_wiki_tunedlens_{i}_blimp_{subject}_good.npy')
    score = all_perp_good < all_perp_bad
    layer_score = np.sum(score, axis=0)
    all_layers_score += layer_score.tolist()
    all_score.append(score)
    # plt.plot(layer_perp)
    # plt.ylabel('Test set avg loss')
    # plt.xlabel('Hidden layer')
    # plt.show()
print(all_layers_score)
# plt.plot(all_layers_score)
# plt.ylabel('Test set avg score')
# plt.xlabel('Hidden layer')
# plt.show()
all_score = np.concatenate(all_score, axis=1)
print(all_score.shape)
np.save(f'score_{subject}_tuned.npy', all_score)

