import numpy as np
import matplotlib.pyplot as plt

all_perp = np.load('results_100000.npy')
layer_perp = np.mean(all_perp, axis=0)
print(layer_perp)
plt.plot(layer_perp)
plt.ylabel('Test set avg loss')
plt.xlabel('Hidden layer')
plt.show()
