"""
USAGE:
    put csv "depth" data to ./data/depth_data
    
    set grid size in m

    run depth.save_to_numpy()

    **Plot kwargs list
        Available kwargs:
        color_bar (bool): show colorbar
        color_range (tuple): range for colormap
        cmap (matplotlib.cmap): cmap for plot
        save_path (str): save path + file name
        fig_size (tuple): figsize
        show_axis (bool): show axis
    **
"""
from processing import depth
from processing.utils.vars import *

box_size_X = DEM_HEIGHT_IN_m
box_size_Z = DEM_WIDTH_IN_m 

# ========== CHECK THIS VARIABLE ===========
grid_size = 10
# ==========================================

grid_resolution_X = int(box_size_X/grid_size)
grid_resolution_Z = int(box_size_Z/grid_size)
grid_shape = (grid_resolution_X, grid_resolution_Z)

print(f"grid shape: {grid_shape}", end="\n\n")

depth.save_to_npy(grid_size, grid_shape, plot=True)
