import numpy as np
from collections import deque

def _bfs_search(arr, start_row, start_col):
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
                cluster = _bfs_search(copied_arr, row, col)
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