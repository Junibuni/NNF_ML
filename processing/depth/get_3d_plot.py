import os
import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd

def get_3d_plot(csv_path):
    """
    Generate a 3D scatter plot of particle data from a CSV file.

    Args:
        csv_path (str): Path to the depth data CSV file.

    Returns:
        None
    """
    # Read CSV data
    data = dd.read_csv(csv_path, skiprows=5, usecols=[1, 2, 3, 4], encoding='latin-1')
    particles = data.compute().to_numpy()

    # Extract data columns
    x = data.iloc[:, 0]
    z = data.iloc[:, 1]
    y = data.iloc[:, 2]
    quantity = data.iloc[:, 3]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with depth-dependent coloring
    sc = ax.scatter(y, x, z, c=quantity, cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Depth')

    # Axis labels
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Example usage:
    csv_path = os.path.join(os.getcwd(), 'depth_data.csv')
    get_3d_plot(csv_path)
