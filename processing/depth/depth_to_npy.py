"""
convert particle depth information to npy format
"""

import glob
import os
import math
import pprint
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd

from pathlib import Path
from time import time
from matplotlib import cm
from tqdm import tqdm

cwd = os.getcwd()
DEPTH_DATA_PATH = os.path.join(cwd, r"processing\data\depth_data")
NUMPY_RESULT_PATH = os.path.join(cwd, r"processing\data\numpy_data")

class Grid:
    def __init__(self, grid_size):
        self.grid_size = grid_size
    def bounding_cell(self, x, z) -> tuple((int, int)):
        # x-coord = matrix columns
        # z-coord = matrix rows

        return (int(z/self.grid_size), int(x/self.grid_size))
    
class Particle:
    def __init__(self, particle_info, xmin, xmax, zmin, zmax):
        _pos_x, _pos_y, _pos_z, _depth = particle_info
        if str(_depth) == 'nan':
            _depth = 0.0
        self.pos_x = _pos_x
        self.pos_y = _pos_y
        self.pos_z = _pos_z
        if xmin < 0.0:
            self.pos_x = _pos_x - xmin
        if zmin < 0.0:
            self.pos_z = _pos_z - zmin
        self.depth = _depth

    def __lt__(self, other):
        return self.depth < other.depth
    
    def __gt__(self, other):
        return self.depth > other.depth

    def __eq__(self, other):
        return self.depth == other.depth
    
    @property
    def height(self):
        return self.depth
    
    def get_position(self):
        return self.pos_x, self.pos_z
    
def get_height(filename, grid_size, grid_shape):
    """
    in:
        filename (str): csv filename with particle depth information
        grid_size (int): grid size in m
        grid_shape (tuple): 2d array shape

    """
    print(f"Processing: {filename}")
    grid = Grid(grid_size)
    id_matrix = [[ [] for i in range(grid_shape[1])] for j in range(grid_shape[0])]
    height = []
    start = time()
    # Read CSV file into a Dask DataFrame
    df = dd.read_csv(filename, sep=",", skiprows=5, usecols=[1, 2, 3, 4], encoding='latin-1')
    particles = df.compute().to_numpy()
    xmin, xmax = min(particles[:,0]), max(particles[:,0])
    zmin, zmax = min(particles[:,2]), max(particles[:,2])
    particles = list(map(lambda x:Particle(x, xmin, xmax, zmin, zmax), particles))
    num_particles = len(particles)

    for particle in tqdm(particles, desc="processing particles", ascii=" *"):
        x, z = particle.get_position()
        row, col = grid.bounding_cell(x, z)
        id_matrix[row][col].append(particle)     
    
    for i,row in enumerate(tqdm(id_matrix, desc="row", ascii=" -", position=0, leave=False)):
        for j,col in enumerate(tqdm(row, desc="column", ascii=" -", position=1, leave=False)):
            maximum = max(id_matrix[i][j]).height if id_matrix[i][j] else 0.0
            height.append(maximum)

    surface_particle = np.array(height).reshape(*grid_shape)
    elp_time = time() - start
    
    print(f"number of particles: {num_particles}")
    print(f"number of grids: {grid_shape[0]*grid_shape[1]}")
    print(f"elapsed time: {elp_time:.2f} sec")
    print()

    del id_matrix
    del grid
    del particles

    return surface_particle

"""
file name format:
강우량_site.csv

ex) 100_12.csv
"""
def save_to_npy(grid_size, grid_shape, plot=False):
    file_lists = glob.glob(DEPTH_DATA_PATH + '\*.csv')
    rainrate_lists = {}
    for file_path in file_lists:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        rain_rate = filename.split("_")[0]
        if rain_rate not in rainrate_lists:
            rainrate_lists[rain_rate] = []
        rainrate_lists[rain_rate].append(file_path)
    print("============================================list of files============================================")
    pprint.PrettyPrinter(indent=4).pprint(rainrate_lists)
    print("=====================================================================================================")
    print()

    for k in rainrate_lists.keys():
        surface_particle = np.zeros(grid_shape)
        save_file_name = os.path.join(NUMPY_RESULT_PATH, k)
        print(f"Processing {k}mm")
        for path in rainrate_lists[k]:
            sf = get_height(path, grid_size, grid_shape)
            surface_particle += sf
        np.save(save_file_name, surface_particle, True)        
        print(f"saved {save_file_name}.npy")
        print()
        
        if plot:
            cmap = cm.coolwarm
            plt.imshow(surface_particle, cmap=cmap)
            #plt.colorbar()
            plt.axis(False)
            #plt.savefig('./full_size_plot.png', bbox_inches='tight')
            plt.show()
