import os
import numpy as np

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

np.savetxt(csv_path, csv_data, delimiter=",")