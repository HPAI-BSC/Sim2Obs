import numpy as np
import matplotlib.pyplot as plt

v = np.load('../maps/Labels/halo4_snap84xyz-zoom1.npz')
print(v['arr_0'])

plt.imshow(v['arr_0'])
plt.show()