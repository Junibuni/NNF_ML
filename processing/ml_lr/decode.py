import os
import pprint
import glob
import numpy as np
import rasterio
import csv
import processing

from processing.utils.vars import *

cwd = os.getcwd()
CHEON_MT_DEM_PATH = os.path.join(cwd, r"processing\data\dem.tif")
NUMPY_RESULT_PATH = os.path.join(cwd, r"processing\data\numpy_data")
CSV_RESULT_PATH = os.path.join(cwd, r"processing\data\csv_data")

def _expand_grid(data, expansion_coefficient):
    expanded_data = []

    for row in data:
        expanded_row = []
        for value in row:
            expanded_row.extend([value] * expansion_coefficient)
        expanded_data.extend([expanded_row] * expansion_coefficient)

    return expanded_data

def index_to_coordinates(row_idx, col_idx, coordinate_data, array_shape):
    num_rows, num_cols = array_shape

    xmin, ymin, xmax, ymax = coordinate_data
    row_step = (ymax - ymin) / num_rows
    col_step = (xmax - xmin) / num_cols

    x_center = xmin + (col_idx * col_step + col_step / 2)
    y_center = ymax - (row_idx * row_step + row_step / 2)

    # return as 위도 경도
    return y_center, x_center

# matrix data to csv data
# get coordinate
# 위도, 경도, 변화높이+지형높이

# get original height map in 461*475 grids
def _get_dem_array(grid_size):
    with rasterio.open(CHEON_MT_DEM_PATH) as src:
        dem_array = src.read(1)

    if BASE_GRID_SIZE_IN_m is not grid_size:
        expansion_coefficient = int(BASE_GRID_SIZE_IN_m / grid_size)
        dem_array = _expand_grid(dem_array, expansion_coefficient)
        print(f"converted grid size from ({BASE_GRID_SIZE_IN_m}m) to ({grid_size}m) : x{expansion_coefficient}")

    return dem_array

def decode(grid_size, depth_array):
    """
    this function decodes numpy 2d array into real coordinate system
    in:
        grid_size (int): grid size in m
        depth_array (np.array): numpy depth array
    out:
        out (np.array): np array stacked information of [x real coordinate, y real coordinate, height information(added with terrain height)]
    """
    out = []

    # load original terrain dem data
    # expand to desired size if needed
    dem_array = _get_dem_array(grid_size)
    array_shape = depth_array.shape #np.array

    for row_idx, row_element in enumerate(depth_array):
        for col_idx, element in enumerate(row_element):
            row_data = []
            if element != 0.0:
                coordinates = index_to_coordinates(row_idx, col_idx, CHEON_MT_COORDINATE, array_shape)
                height_total = dem_array[row_idx][col_idx] + element
                row_data = [*coordinates, height_total]
                out.append(row_data)
    print("decoded")
    return np.array(out)
