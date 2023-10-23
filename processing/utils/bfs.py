import numpy as np
from collections import deque

def bfs_search(arr, start_row, start_col):
    """
    Perform a breadth-first search (BFS) to find a connected cluster in a NumPy array.

    Args:
        arr (np.array): Input NumPy array.
        start_row (int): Starting row index for BFS.
        start_col (int): Starting column index for BFS.

    Returns:
        set: A set containing the indices of the cluster.
    """
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

def remove_small_clusters(arr, threshold=150, label_true=None, label_false=None):
    """
    Remove small clusters from a NumPy array.

    Args:
        arr (np.array): Input NumPy array.
        threshold (int): Minimum size of clusters to keep.
        label_true: Label to assign to valid clusters (optional).
        label_false: Label to assign to removed clusters (optional).

    Returns:
        np.array: Modified array with small clusters removed.
    """
    clusters = []
    not_clustered = []
    arr_copy = arr.copy()
    
    for row in range(arr_copy.shape[0]):
        for col in range(arr_copy.shape[1]):
            if arr_copy[row, col] != 0.0:
                cluster = bfs_search(arr_copy, row, col)
                if len(cluster) >= threshold:
                    clusters.append(cluster)
                else:
                    not_clustered.append(cluster)

    new_arr = arr.copy()
    
    if label_true:
        for cluster in clusters:
            for r, c in cluster:
                new_arr[r, c] = label_true

    for cluster in not_clustered:
        for r, c in cluster:
            if label_false:
                new_arr[r, c] = label_false
            else:
                new_arr[r, c] = 0.0

    return new_arr
