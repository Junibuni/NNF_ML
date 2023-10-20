import numpy as np

from processing.utils.bfs import remove_small_clusters

# Example
import os
import glob
import matplotlib.pyplot as plt

"""
clustered_data = np.zeros((300, 300))
num_clusters = 50

max_cluster_size = 600

for cluster_id in range(1, num_clusters + 1):
    cluster_size = np.random.randint(10, max_cluster_size)
    
    center_x = np.random.randint(0, 300)
    center_y = np.random.randint(0, 300)

    for _ in range(cluster_size):
        forX = np.random.randint(-1, 2)
        forY = 0
        if forX == 0:
            forY = np.random.choice([-1, 1])

        x = center_x + forX
        y = center_y + forY
        
        x = max(0, min(299, x))
        y = max(0, min(299, y))
        
        if clustered_data[x, y] == 0.0:
            if cluster_size < threshold:
                clustered_data[x, y] = -1
            else:
                clustered_data[x, y] = 1
        center_x, center_y = x, y
    """

threshold = 500

cwd = os.getcwd()
file_path = os.path.join(cwd, "processing/data/numpy_data/150.npy")
data = np.load(file_path)

vmin = -2
vmax = 2

result = remove_small_clusters(data, threshold=threshold, label_true=300, label_false=-200)
plt.imshow(data, vmin=vmin, vmax=vmax)
plt.axis(False)
plt.show()

plt.imshow(result, vmin=vmin, vmax=vmax)
plt.axis(False)
plt.show()

print("Original Array:")
print(data)
print("Array with Small Clusters Removed:")
print(result)

