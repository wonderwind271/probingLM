import numpy as np
import matplotlib.pyplot as plt

all_layers_perp = []

for i in range(4):
    all_perp = np.load(f'results_wiki_tunedlens_{i}.npy')
    layer_perp = np.mean(all_perp, axis=0)
    all_layers_perp += layer_perp.tolist()
    # plt.plot(layer_perp)
    # plt.ylabel('Test set avg loss')
    # plt.xlabel('Hidden layer')
    # plt.show()
print(all_layers_perp)
plt.plot(all_layers_perp)
plt.ylabel('Test set avg loss')
plt.xlabel('Hidden layer')
plt.show()