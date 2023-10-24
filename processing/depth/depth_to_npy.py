import os
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
import math
import pprint
import glob
from tqdm import tqdm

# Define constants
cwd = os.getcwd()
DEPTH_DATA_PATH = os.path.join(cwd, "processing/data/depth_data")
NUMPY_RESULT_PATH = os.path.join(cwd, "processing/data/numpy_data")

# Define Grid class
class Grid:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def bounding_cell(self, x, z):
        # Calculate the grid cell based on coordinates
        return int(z / self.grid_size), int(x / self.grid_size)

# Define Particle class
class Particle:
    def __init__(self, particle_info, xmin, xmax, zmin, zmax):
        pos_x, pos_y, pos_z, depth = particle_info
        if math.isnan(depth):
            depth = 0.0
        self.pos_x = pos_x - xmin if xmin < 0.0 else pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z - zmin if zmin < 0.0 else pos_z
        self.depth = depth

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

# Define a function to get height data
def get_height(filename, grid_size, grid_shape):
    print('\033[37m' + f"Processing: {filename}")
    grid = Grid(grid_size)
    id_matrix = [[[] for _ in range(grid_shape[1])] for _ in range(grid_shape[0])]
    height = []

    # Read CSV file into a Dask DataFrame
    df = dd.read_csv(filename, sep=",", skiprows=5, usecols=[1, 2, 3, 4], encoding='latin-1')
    particles = df.compute().to_numpy()
    xmin, xmax = min(particles[:, 0]), max(particles[:, 0])
    zmin, zmax = min(particles[:, 2]), max(particles[:, 2])
    particles = [Particle(particle, xmin, xmax, zmin, zmax) for particle in particles]

    for particle in tqdm(particles, desc="Processing particles"):
        x, z = particle.get_position()
        row, col = grid.bounding_cell(x, z)
        id_matrix[row][col].append(particle)

    for i, row in enumerate(tqdm(id_matrix, desc="Row", position=0, leave=False)):
        for j, col in enumerate(tqdm(row, desc="Column", position=1, leave=False)):
            maximum = max(id_matrix[i][j], key=lambda p: p.height).height if id_matrix[i][j] else 0.0
            height.append(maximum)

    surface_particle = np.array(height).reshape(*grid_shape)

    del id_matrix
    del grid
    del particles

    return surface_particle

# Define a function to plot data
def plot_data(data, **kwargs):
    color_bar = kwargs.get("color_bar", False)
    color_range = kwargs.get("color_range", None)
    cmap = kwargs.get("cmap", plt.cm.coolwarm)
    save_path = kwargs.get("save_path", None)
    fig_size = kwargs.get("fig_size", None)
    show_axis = kwargs.get("show_axis", False)

    plt.figure(figsize=fig_size)
    plt.imshow(data, cmap=cmap)

    if color_bar:
        plt.colorbar()
    if color_range is not None:
        plt.clim(*color_range)

    plt.axis('on' if show_axis else 'off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved as {save_path}")

    plt.show()
    plt.close()

# Define a function to save data to npy files
def save_to_npy(grid_size, grid_shape, plot=False, num_site=2, filepath=None, **kwargs):
    if filepath:
        save_file_name = os.path.join(NUMPY_RESULT_PATH, "test")
        sf = get_height(filepath, grid_size, grid_shape)
        np.save(save_file_name, sf, allow_pickle=True)
        print(f"Saved {save_file_name}.npy", end="\n\n")
        if plot:
            plot_data(sf, **kwargs)
        return
    file_lists = glob.glob(os.path.join(DEPTH_DATA_PATH, '*.csv'))
    rainrate_lists = {}
    
    for file_path in file_lists:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        rain_rate = filename.split("_")[0]
        rainrate_lists.setdefault(rain_rate, []).append(file_path)

    print("============================================ List of Files ============================================")
    pprint.pprint(rainrate_lists)
    print("=======================================================================================================")

    for k, file_paths in rainrate_lists.items():
        if len(file_paths) != num_site:
            print(f"Skipping {k}mm due to the number of files for {k}mm less than {num_site}")
            continue

        surface_particle = np.zeros(grid_shape)
        save_file_name = os.path.join(NUMPY_RESULT_PATH, k)
        print('\033[34m' + f"Processing {k}mm")

        for path in file_paths:
            sf = get_height(path, grid_size, grid_shape)
            surface_particle += sf

        np.save(save_file_name, surface_particle, allow_pickle=True)
        print(f"Saved {save_file_name}.npy", end="\n\n")

        if plot:
            plot_data(surface_particle, rain_rate=k, **kwargs)

# Example usage
if __name__ == "__main__":
    grid_size = 10
    grid_shape = (461, 475)
    save_to_npy(grid_size, grid_shape, plot=True)
