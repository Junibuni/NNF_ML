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

def label_clusters(input_array):
    """
    Label clusters in a 2D NumPy array.

    Args:
        input_array (numpy.ndarray): The input 2D array containing float values.

    Returns:
        cluster_map (numpy.ndarray): A labeled cluster map.
        num_clusters (int): The total number of identified clusters.
    """
    def bfs_label(cluster_map, i, j, label):
        q = deque([(i, j)])
        while q:
            x, y = q.popleft()
            if 0 <= x < rows and 0 <= y < cols and input_array[x, y] != 0 and cluster_map[x, y] == 0:
                cluster_map[x, y] = label
                q.extend([(x + dx, y + dy) for dx, dy in directions])

    rows, cols = input_array.shape
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    cluster_map = np.zeros((rows, cols), dtype=int)
    label = 0

    for i in range(rows):
        for j in range(cols):
            if input_array[i, j] != 0 and cluster_map[i, j] == 0:
                label += 1
                bfs_label(cluster_map, i, j, label)

    return cluster_map, label

def extract_cluster(cluster_map, target_label, original_array):
    """
    Extract a specific labeled cluster from a cluster map.

    Args:
        cluster_map (numpy.ndarray): The labeled cluster map.
        target_label (int): The label of the cluster to extract.

    Returns:
        extracted_cluster (numpy.ndarray): The extracted cluster with 1s representing the target label and 0s elsewhere.
    """
    return np.where(cluster_map == target_label, original_array, 0)

# Example usage
if __name__ == '__main__':
    input_array = np.array([[0.3, 0.4, 0.0, 0.0, 0.0],
                            [0.8, 0.9, 0.0, 0.6, 0.7],
                            [0.0, 0.0, 0.0, 0.5, 0.0],
                            [0.2, 0.1, 0.0, 0.0, 0.4]])

    cluster_map, num_clusters = label_clusters(input_array)

    # Extract the 2nd cluster (change the label as needed)
    target_label = 2
    extracted_cluster = extract_cluster(cluster_map, target_label, input_array)

    print("Cluster Map:")
    print(cluster_map)

    print(f"Extracted Cluster {target_label}:")
    print(extracted_cluster)