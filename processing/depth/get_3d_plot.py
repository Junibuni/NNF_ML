import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd

from mpl_toolkits.mplot3d import Axes3D

cwd = os.getcwd()

def get_3d_plot(csv_path):
    """
    in:
        csv_path (str): depth data from SPH simulation (csv)
    out:
        None
    """
    data = dd.read_csv(csv_path, skiprows=5, usecols=[1, 2, 3, 4], encoding='latin-1')
    particles = data.compute().to_numpy()
    x = data.iloc[:, 0]
    z = data.iloc[:, 1]
    y = data.iloc[:, 2]
    quantity = data.iloc[:, 3]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(y, x, z, c=quantity, cmap='viridis')

    cbar = plt.colorbar(sc)
    cbar.set_label('Depth')

    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    plt.show()