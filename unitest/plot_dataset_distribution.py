import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F

def plot_distribution(dataset_path):
    """
    This function plots the distribution of forward_transmission as a histogram 
    and uses T-SNE to visualize the dataset with colors representing different 
    levels of forward_transmission.

    Parameters:
    - dataset_path (str): Path to the dataset folder containing 1230 .h5 files.
    """
    # Initialize lists to store data and labels
    data = []
    labels = []
    dataset_name = dataset_path.split("/")[-1]
    # Iterate through all .h5 files in the dataset path
    for filename in os.listdir(dataset_path):
        if filename.endswith(".h5"):
            file_path = os.path.join(dataset_path, filename)
            with h5py.File(file_path, 'r') as f:
                if "perturb" in file_path:
                    step = file_path.split("_opt_step_")[-1].split("_perturb")[0]
                else:
                    step = file_path.split("_opt_step_")[-1].split(".")[0]
                # breakdown_fwd_trans_value_step-0
                target_size = (96, 96)
                design_region_x_start = f["design_region_mask-bending_region_x_start"][()]
                design_region_x_stop = f["design_region_mask-bending_region_x_stop"][()]
                design_region_y_start = f["design_region_mask-bending_region_y_start"][()]
                design_region_y_stop = f["design_region_mask-bending_region_y_stop"][()]
                eps_map = np.array(f["eps_map"])[
                    design_region_x_start:design_region_x_stop,
                    design_region_y_start:design_region_y_stop
                ]  # Load the 2D tensor
                eps_map = torch.tensor(eps_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                eps_map = F.interpolate(eps_map, size=target_size, mode='bilinear', align_corners=True)
                eps_map = eps_map.squeeze().numpy()
                forward_transmission = float(f[f"breakdown_fwd_trans_value_step-{step}"][()])  # Load the label
                data.append(eps_map.flatten())  # Flatten 2D tensor to 1D
                labels.append(forward_transmission)

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Plot 1: Histogram of forward_transmission
    plt.figure(figsize=(10, 5))
    bins = np.linspace(0, 1, 10)  # Self-defined bins
    plt.hist(labels, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Histogram of Forward Transmission")
    plt.xlabel("Forward Transmission")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(f"./figs/histogram_{dataset_name}.png")

    # Plot 2: T-SNE visualization
    # Normalize data for T-SNE
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    # Apply T-SNE
    # 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(normalized_data)

    # Create a scatter plot with labels as colors
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=10, alpha=0.8)
    plt.colorbar(scatter, label="Forward Transmission")
    plt.title("T-SNE Visualization of Dataset")
    plt.xlabel("T-SNE Component 1")
    plt.ylabel("T-SNE Component 2")
    plt.grid(True)
    plt.savefig(f"./figs/tsne_{dataset_name}_2d.png", dpi=300)
    # 3D
    # Apply t-SNE with 3D components
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    tsne_results_3d = tsne.fit_transform(data)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with color representing forward_transmission
    scatter = ax.scatter(
        tsne_results_3d[:, 0], 
        tsne_results_3d[:, 1], 
        tsne_results_3d[:, 2], 
        c=labels, cmap='viridis', s=10, alpha=0.8
    )

    # Add colorbar and labels
    cbar = plt.colorbar(scatter, ax=ax, pad=0.2)
    cbar.set_label("Forward Transmission")
    ax.set_title("3D t-SNE Visualization of Dataset")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_zlabel("t-SNE Component 3")
    plt.savefig(f"./figs/tsne_{dataset_name}_3d.png", dpi=300)

if __name__ == "__main__":

    dataset_dir = "./data/fdfd/bending/raw_opt_traj_10_ptb"

    plot_distribution(dataset_dir)