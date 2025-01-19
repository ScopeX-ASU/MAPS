# import os
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import MinMaxScaler
# import torch
# import torch.nn.functional as F
# import csv

# def plot_distribution(dataset_path):
#     """
#     This function plots the distribution of forward_transmission as a histogram 
#     and uses T-SNE to visualize the dataset with colors representing different 
#     levels of forward_transmission.

#     Parameters:
#     - dataset_path (str): Path to the dataset folder containing 1230 .h5 files.
#     """
#     # Initialize lists to store data and labels
#     data = []
#     labels = []
#     dataset_name = dataset_path.split("/")[-1]
#     # Iterate through all .h5 files in the dataset path
#     for filename in os.listdir(dataset_path):
#         if filename.endswith(".h5"):
#             file_path = os.path.join(dataset_path, filename)
#             with h5py.File(file_path, 'r') as f:
#                 # else:
#                 #     step = file_path.split("_opt_step_")[-1].split(".")[0]
#                 # breakdown_fwd_trans_value_step-0
#                 target_size = (96, 96)
#                 design_region_x_start = f["design_region_mask-bending_region_x_start"][()]
#                 design_region_x_stop = f["design_region_mask-bending_region_x_stop"][()]
#                 design_region_y_start = f["design_region_mask-bending_region_y_start"][()]
#                 design_region_y_stop = f["design_region_mask-bending_region_y_stop"][()]
#                 eps_map = np.array(f["eps_map"])[
#                     design_region_x_start:design_region_x_stop,
#                     design_region_y_start:design_region_y_stop
#                 ]  # Load the 2D tensor
#                 eps_map = torch.tensor(eps_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#                 eps_map = F.interpolate(eps_map, size=target_size, mode='bilinear', align_corners=True)
#                 eps_map = eps_map.squeeze().numpy()
#                 if "breakdown_fwd_trans_value" not in list(f.keys()):
#                     continue
#                 forward_transmission = float(f[f"breakdown_fwd_trans_value"][()])  # Load the label
#                 data.append(eps_map.flatten())  # Flatten 2D tensor to 1D
#                 labels.append(forward_transmission)
#     print("this is the length of the data", len(data))
#     # Convert lists to numpy arrays
#     data = np.array(data)
#     labels = np.array(labels)

#     # Plot 2: T-SNE visualization
#     # Normalize data for T-SNE
#     scaler = MinMaxScaler()
#     normalized_data = scaler.fit_transform(data)
#     # quit()
#     # Apply T-SNE
#     # 2D
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30)
#     tsne_results = tsne.fit_transform(normalized_data)

#     # Create a scatter plot with labels as colors
#     # plt.figure(figsize=(10, 7))
#     # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=10, alpha=0.8)
#     # plt.colorbar(scatter, label="Forward Transmission")
#     # plt.title("T-SNE Visualization of Dataset")
#     # plt.xlabel("T-SNE Component 1")
#     # plt.ylabel("T-SNE Component 2")
#     # plt.grid(True)
#     # plt.savefig(f"./figs/tsne_{dataset_name}_2d.png", dpi=300)
#     # # 3D
#     # # Apply t-SNE with 3D components
#     # tsne = TSNE(n_components=3, random_state=42, perplexity=30)
#     # tsne_results_3d = tsne.fit_transform(data)

#     # # Create a 3D scatter plot
#     # fig = plt.figure(figsize=(10, 7))
#     # ax = fig.add_subplot(111, projection='3d')

#     # # Scatter plot with color representing forward_transmission
#     # scatter = ax.scatter(
#     #     tsne_results_3d[:, 0], 
#     #     tsne_results_3d[:, 1], 
#     #     tsne_results_3d[:, 2], 
#     #     c=labels, cmap='viridis', s=10, alpha=0.8
#     # )

#     # # Add colorbar and labels
#     # cbar = plt.colorbar(scatter, ax=ax, pad=0.2)
#     # cbar.set_label("Forward Transmission")
#     # ax.set_title("3D t-SNE Visualization of Dataset")
#     # ax.set_xlabel("t-SNE Component 1")
#     # ax.set_ylabel("t-SNE Component 2")
#     # ax.set_zlabel("t-SNE Component 3")
#     # plt.savefig(f"./figs/tsne_{dataset_name}_3d.png", dpi=300)

#     # # ----
#     # # Plot 3D t-SNE visualization with transparent background
#     # # Apply t-SNE with 3D components
#     # tsne = TSNE(n_components=3, random_state=42, perplexity=30)
#     # tsne_results_3d = tsne.fit_transform(data)

#     # # Create a 3D scatter plot
#     # fig = plt.figure(figsize=(10, 7))
#     # ax = fig.add_subplot(111, projection='3d')

#     # # Scatter plot with color representing forward_transmission
#     # ax.scatter(
#     #     tsne_results_3d[:, 0], 
#     #     tsne_results_3d[:, 1], 
#     #     tsne_results_3d[:, 2], 
#     #     c=labels, cmap='viridis', s=10, alpha=0.8
#     # )

#     # # Set transparent background for the 3D plot
#     # ax.set_facecolor('none')  # Set the axes' background to transparent
#     # fig.patch.set_alpha(0)    # Set the figure's patch to transparent
#     # ax.grid(True)  # Optionally retain the grid

#     # # Add labels
#     # ax.set_xlabel("t-SNE Component 1")
#     # ax.set_ylabel("t-SNE Component 2")
#     # ax.set_zlabel("t-SNE Component 3")

#     # # Save the figure with a transparent background
#     # plt.savefig(f"./figs/tsne_{dataset_name}_3d.png", dpi=300, transparent=True)

#     # Plot 3D t-SNE visualization and save scatter points and background separately
#     # Apply t-SNE with 3D components
#     tsne = TSNE(n_components=3, random_state=42, perplexity=30)
#     tsne_results_3d = tsne.fit_transform(data)

#     # Save scatter points only (transparent background)
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')

#     # Scatter plot with color representing forward_transmission
#     scatter = ax.scatter(
#         tsne_results_3d[:, 0], 
#         tsne_results_3d[:, 1], 
#         tsne_results_3d[:, 2], 
#         c=labels, cmap='viridis', s=10, alpha=0.8
#     )

#     # Set transparent background
#     ax.set_facecolor('none')  # Transparent axes background
#     fig.patch.set_alpha(0)    # Transparent figure background
#     ax.grid(False)            # Turn off grid for scatter points
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     ax.set_zlabel("")
#     plt.axis('off')           # Remove axes for scatter points

#     # Adjust layout to minimize margins
#     plt.tight_layout(pad=0)

#     # Save scatter points
#     plt.savefig(f"./figs/tsne_{dataset_name}_3d_scatter.png", dpi=300, transparent=True, bbox_inches='tight')
#     plt.close(fig)  # Close the figure to reset

#     # Save background only (no scatter points)
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')

#     # Background with axis and grid
#     ax.set_facecolor('white')  # Opaque background for the axes
#     fig.patch.set_alpha(1)     # Opaque figure background
#     ax.grid(True)              # Retain grid for background
#     ax.set_xlabel("t-SNE Component 1")
#     ax.set_ylabel("t-SNE Component 2")
#     ax.set_zlabel("t-SNE Component 3")

#     # Adjust layout to minimize margins
#     plt.tight_layout(pad=0)

#     # Save the background only
#     plt.savefig(f"./figs/tsne_{dataset_name}_3d_background.png", dpi=300, transparent=False, bbox_inches='tight')
#     plt.close(fig)  # Close the figure to reset



# if __name__ == "__main__":

#     dataset_dir = "./data/fdfd/bending/raw_opt_traj_ptb"
#     # dataset_dir = "./data/fdfd/bending/raw_random"

#     plot_distribution(dataset_dir)



import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F

def plot_combined_tsne_3d(dataset_path1, dataset_path2):
    """
    This function plots the combined 3D t-SNE visualization for datasets in two folders.
    Data from different folders are distinguished using different color maps.

    Parameters:
    - dataset_path1 (str): Path to the first dataset folder.
    - dataset_path2 (str): Path to the second dataset folder.
    """
    def load_data(dataset_path):
        """Helper function to load data and labels from a dataset folder."""
        data = []
        labels = []
        for filename in os.listdir(dataset_path):
            if filename.endswith(".h5"):
                file_path = os.path.join(dataset_path, filename)
                with h5py.File(file_path, 'r') as f:
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
                    if "breakdown_fwd_trans_value" not in list(f.keys()):
                        continue
                    forward_transmission = float(f[f"breakdown_fwd_trans_value"][()])  # Load the label
                    data.append(eps_map.flatten())  # Flatten 2D tensor to 1D
                    labels.append(forward_transmission)
        return np.array(data), np.array(labels)

    # Load data from both datasets
    data1, labels1 = load_data(dataset_path1)
    data2, labels2 = load_data(dataset_path2)

    # Normalize data
    scaler = MinMaxScaler()
    normalized_data1 = scaler.fit_transform(data1)
    normalized_data2 = scaler.fit_transform(data2)

    # Combine data for t-SNE
    combined_data = np.vstack([normalized_data1, normalized_data2])
    dataset_labels = np.array([0] * len(data1) + [1] * len(data2))  # 0 for dataset1, 1 for dataset2

    # Apply t-SNE (3D)
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(combined_data)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot dataset 1
    scatter1 = ax.scatter(
        tsne_results[dataset_labels == 0, 0],
        tsne_results[dataset_labels == 0, 1],
        tsne_results[dataset_labels == 0, 2],
        c=labels1,
        cmap='viridis',
        s=10,
        alpha=0.8,
        label="Puerturbed Opt-Traj Sampling"
    )

    # Plot dataset 2
    scatter2 = ax.scatter(
        tsne_results[dataset_labels == 1, 0],
        tsne_results[dataset_labels == 1, 1],
        tsne_results[dataset_labels == 1, 2],
        c=labels2,
        cmap='plasma',
        s=10,
        alpha=0.8,
        label="Random Sampling"
    )

    # Add legend, colorbars, and labels
    cbar1 = plt.colorbar(scatter1, ax=ax, pad=0.1, location='left', shrink=0.6)
    cbar1.set_label("Puerturbed Opt-Traj Sampling Transmission")
    cbar2 = plt.colorbar(scatter2, ax=ax, pad=0.1, location='right', shrink=0.6)
    cbar2.set_label("Random Sampling Forward Transmission")

    ax.set_title("3D t-SNE Visualization of Two Datasets")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_zlabel("t-SNE Component 3")
    ax.legend()

    # Save the plot
    combined_name = f"{dataset_path1.split('/')[-1]}_vs_{dataset_path2.split('/')[-1]}"
    plt.savefig(f"./figs/tsne_3d_combined_{combined_name}.png", dpi=300)

if __name__ == "__main__":
    dataset_dir1 = "./data/fdfd/bending/raw_opt_traj_ptb"
    dataset_dir2 = "./data/fdfd/bending/raw_random"
    plot_combined_tsne_3d(dataset_dir1, dataset_dir2)
