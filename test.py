import numpy as np
from collections import deque

def bfs_search(arr, start_row, start_col):
    queue = deque([(start_row, start_col)])
    cluster = set()

    while queue:
        row, col = queue.popleft()
        if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1] and arr[row, col] != 0.0:
            cluster.add((row, col))
            arr[row, col] = 0.0  # Mark as visited

            neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            for r, c in neighbors:
                if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1] and (r, c) not in cluster:
                    queue.append((r, c))

    return cluster

def remove_small_clusters(arr, threshold=150):
    copied_arr = arr.copy()
    clusters = []
    not_clusetered = []

    for row in range(copied_arr.shape[0]):
        for col in range(copied_arr.shape[1]):
            if copied_arr[row, col] != 0.0:
                cluster = bfs_search(copied_arr, row, col)
                if len(cluster) >= threshold:
                    clusters.append(cluster)
                else:
                    not_clusetered.append(cluster)
    
    new_arr = np.zeros(arr.shape)
    for cluster in clusters:
        for r, c in cluster:
            new_arr[r, c] = 1.0
    
    for cluster in not_clusetered:
        for r, c, in cluster:
            new_arr[r, c] = -2.0
    
    return new_arr

# Example
import os
import glob
import matplotlib.pyplot as plt
threshold = 500

cwd = os.getcwd()
file_path = os.path.join(cwd, "processing/data/numpy_data/250.npy")
data = np.load(file_path)
result = remove_small_clusters(data)
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

vmin = -2
vmax = 2

result = remove_small_clusters(data, threshold=threshold)
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

