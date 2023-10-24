import numpy as np
import glob
import os
import matplotlib.pyplot as plt

from processing.utils.bfs import remove_small_clusters
from processing.utils.bfs import label_clusters, extract_cluster
from processing import ml_lr
from processing.utils.vars import *

# ========== CHECK THIS VARIABLE ===========
grid_size = 10
# ==========================================

cwd = os.getcwd()
npy_path = os.path.join(cwd, r"processing\data\numpy_data\150.npy")
csv_path = os.path.join(cwd, r"processing\data\coordinate_data\fixed_final.csv")

depth_array = np.load(npy_path)
csv_data = ml_lr.decode(grid_size, depth_array)

result = remove_small_clusters(depth_array)

cluster_map, num_clusters = label_clusters(result)

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

for i in range(1, 4):
    target_label = i
    save_csv = os.path.join(cwd, fr"processing\data\coordinate_data\150_{i}.csv")

    extracted_cluster = extract_cluster(cluster_map, target_label, result)
    csv_data = ml_lr.decode(grid_size, extracted_cluster)
    np.savetxt(save_csv, csv_data, delimiter=",")