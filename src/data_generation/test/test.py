import numpy as np
import matplotlib.pyplot as plt

v = np.load('../data/npz/labels/halo-3_subhalo-442_snap84xyz-zoom1.npz')
print(v['arr_0'])

plt.imshow(v['arr_0'])
plt.show()