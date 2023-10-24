import numpy as np
import glob
import os
import matplotlib.pyplot as plt

from processing.utils.bfs import remove_small_clusters
from processing.utils.bfs import label_clusters, extract_cluster

cwd = os.getcwd()
file_path = os.path.join(cwd, "processing/data/numpy_data/150.npy")
data = np.load(file_path)
result = remove_small_clusters(data)

cluster_map, num_clusters = label_clusters(result)

# Extract the 2nd cluster (change the label as needed)
target_label = 2
extracted_cluster = extract_cluster(cluster_map, target_label, result)

print("Cluster Map:")
print(cluster_map)
plt.imshow(cluster_map)
plt.axis(False)
plt.show()

print(f"Extracted Cluster {target_label}:")
print(extracted_cluster)
plt.imshow(extracted_cluster)
plt.axis(False)
plt.show()
quit()
# Example
import os
import glob
import matplotlib.pyplot as plt

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

