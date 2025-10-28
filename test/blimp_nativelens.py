import numpy as np
import matplotlib.pyplot as plt

all_layers_score = []
subject = 'animate_subject_passive'

all_perp_bad = np.load(f'result_blimp_nativelens/results_wiki_nativelens_blimp_{subject}_bad.npy')
all_perp_good = np.load(f'result_blimp_nativelens/results_wiki_nativelens_blimp_{subject}_good.npy')
score = all_perp_good < all_perp_bad
score = score[:, :-1]
layer_score = np.sum(score, axis=0)


all_layers_score += layer_score.tolist()
    # plt.plot(layer_perp)
    # plt.ylabel('Test set avg loss')
    # plt.xlabel('Hidden layer')
    # plt.show()
print(all_layers_score)
# plt.plot(all_layers_score)
# plt.ylabel('Test set avg score')
# plt.xlabel('Hidden layer')
# plt.show()
np.save(f'score_{subject}_native.npy', score)
