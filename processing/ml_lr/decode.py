import os
import glob
import numpy as np
import rasterio
from processing.utils.vars import *

cwd = os.getcwd()
CHEON_MT_DEM_PATH = os.path.join(cwd, "processing/data/dem.tif")
NUMPY_RESULT_PATH = os.path.join(cwd, "processing/data/numpy_data")
CSV_RESULT_PATH = os.path.join(cwd, "processing/data/coordinate_data")

def expand_grid(data, expansion_coefficient):
    """
    Expand a grid by repeating values in each cell.

    Args:
        data (list of lists): Input grid data.
        expansion_coefficient (int): Factor by which to expand the grid.

    Returns:
        list of lists: Expanded grid data.
    """
    expanded_data = []

    for row in data:
        expanded_row = []
        for value in row:
            expanded_row.extend([value] * expansion_coefficient)
        expanded_data.extend([expanded_row] * expansion_coefficient)

    return expanded_data

def index_to_coordinates(row_idx, col_idx, coordinate_data, array_shape):
    """
    Convert grid indices to real-world coordinates.

    Args:
        row_idx (int): Row index in the grid.
        col_idx (int): Column index in the grid.
        coordinate_data (tuple): Tuple of (xmin, ymin, xmax, ymax).
        array_shape (tuple): Tuple of (num_rows, num_cols).

    Returns:
        tuple: Real-world coordinates (latitude, longitude).
    """
    num_rows, num_cols = array_shape
    xmin, ymin, xmax, ymax = coordinate_data

    row_step = (ymax - ymin) / num_rows
    col_step = (xmax - xmin) / num_cols

    x_center = xmin + (col_idx * col_step + col_step / 2)
    y_center = ymax - (row_idx * row_step + row_step / 2)

    return y_center, x_center

def get_dem_array(grid_size):
    """
    Get the DEM array and optionally expand it to the desired grid size.

    Args:
        grid_size (int): Desired grid size in meters.

    Returns:
        np.array: DEM array.
    """
    with rasterio.open(CHEON_MT_DEM_PATH) as src:
        dem_array = src.read(1)

    if BASE_GRID_SIZE_IN_m != grid_size:
        expansion_coefficient = int(BASE_GRID_SIZE_IN_m / grid_size)
        dem_array = expand_grid(dem_array, expansion_coefficient)
        print(f"Converted grid size from ({BASE_GRID_SIZE_IN_m}m) to ({grid_size}m): x{expansion_coefficient}")

    return dem_array

def decode(grid_size, depth_array):
    """
    Decode a numpy 2D array into real coordinate system.

    Args:
        grid_size (int): Grid size in meters.
        depth_array (np.array): Numpy depth array.

    Returns:
        np.array: Numpy array with stacked information of [x real coordinate, y real coordinate, height information (added with terrain height)].
    """
    out = []
    dem_array = get_dem_array(grid_size)
    array_shape = depth_array.shape

    for row_idx, row_element in enumerate(depth_array):
        for col_idx, element in enumerate(row_element):
            if element != 0.0:
                coordinates = index_to_coordinates(row_idx, col_idx, CHEON_MT_COORDINATE, array_shape)
                height_total = dem_array[row_idx][col_idx] + element
                row_data = [*coordinates, height_total]
                out.append(row_data)
    print("Decoded")
    return np.array(out)

# Example usage
if __name__ == "__main__":
    BASE_GRID_SIZE_IN_m = 10 

    xmin, ymin, xmax, ymax = 127.959666, 37.073320, 128.019327, 37.115297
    CHEON_MT_COORDINATE = (xmin, ymin, xmax, ymax)
    grid_size = 10

    cwd = os.getcwd()
    npy_path = os.path.join(cwd, r"processing\data\numpy_data\150.npy")
    csv_path = os.path.join(cwd, r"processing\data\coordinate_data\fixed_final.csv")

    depth_array = np.load(npy_path)
    decoded_data = decode(grid_size, depth_array)
    print(decoded_data)
    np.savetxt(csv_path, decoded_data, delimiter=",")