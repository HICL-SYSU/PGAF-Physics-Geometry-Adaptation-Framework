# predict.py 
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import copy  # Used for deep copying data objects to avoid side effects

# Import project-specific modules
import config
from model import PGAF
from mydata import MultiFolderDataset
from torch_geometric.data import DataLoader


def visualize_field(fig, subplot_index, pos, field_data, title, cmap='jet', v_min=None, v_max=None):
    """
    A general helper function to plot a 3D scalar field on a given figure's subplot.

    Args:
        fig (matplotlib.figure.Figure): The Matplotlib figure object to draw on.
        subplot_index (int): The subplot index (e.g., 121 for a 1x2 grid, 1st plot).
        pos (np.ndarray): Coordinates of the points, shape (N, 3).
        field_data (np.ndarray): Scalar value at each point, shape (N,).
        title (str): The title for the subplot.
        cmap (str): The colormap to use for visualization.
        v_min, v_max (float, optional): The range for the color bar. If None, it's determined automatically.
    """
    ax = fig.add_subplot(subplot_index, projection='3d')
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=field_data, cmap=cmap, s=5, vmin=v_min, vmax=v_max)
    ax.set_title(title)
    # Set a consistent viewing angle for better comparison
    ax.view_init(elev=30, azim=45)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.colorbar(sc, ax=ax, shrink=0.6)
    return ax


def main():
    """
    Main execution function for prediction and visualization.
    """
    print(f"--- Final Model Evaluation on the Test Set ---")
    print(f"Loading best model from: {config.MODEL_SAVE_PATH}")
    device = config.DEVICE

    # 1. Load the test dataset and the trained model
    # ----------------------------------------------------
    test_dataset = MultiFolderDataset(root_dir=config.DATA_DIR, split='test')
    if len(test_dataset) == 0:
        print("Test dataset is empty. Cannot perform final evaluation. Please populate the 'test' directory.")
        return
    # Use a DataLoader with batch_size=1 to process one sample at a time
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, follow_batch=['query_pos'])

    model = PGAF().to(device)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Model file not found at {config.MODEL_SAVE_PATH}. Please train the model first.")
        return
    # Load the saved model state and set the model to evaluation mode
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # 2. Prepare the input for inference
    # ----------------------------------------------------
    print("Processing the first sample from the test set...")
    # Get the first (and only, due to batch_size=1) sample from the loader
    original_data_sample = next(iter(test_loader))

    # To get predictions on the entire geometry (boundary + interior),
    # we create a new data object for inference.
    inference_data = copy.deepcopy(original_data_sample)
    
    # Concatenate boundary and interior points to form the full set of query points
    all_points_pos_tensor = torch.cat([inference_data.pos, inference_data.query_pos], dim=0)
    inference_data.query_pos = all_points_pos_tensor
    
    # A new batch index vector is required for the new query_pos
    inference_data.query_pos_batch = torch.zeros(all_points_pos_tensor.size(0), dtype=torch.long)
    
    # Move the inference data to the target device
    inference_data = inference_data.to(device)

    # 3. Run inference
    # ----------------------------------------------------
    print("Running inference...")
    with torch.no_grad():  # Disable gradient calculation for efficiency
        # The CVAE model is generative, so each call will produce a slightly different result.
        # We run it once for this visualization.
        predicted_solution_all_points = model(inference_data)

    print("Prediction complete. Preparing data for visualization...")

    # 4. Prepare data for plotting
    # ----------------------------------------------------
    # Ground Truth data
    true_pos_all = torch.cat([original_data_sample.pos, original_data_sample.query_pos], dim=0).cpu().numpy()
    true_flow_all = torch.cat([original_data_sample.x, original_data_sample.y], dim=0).cpu().numpy()
    
    # Predicted data
    pred_pos_all = inference_data.query_pos.cpu().numpy()
    pred_flow_all = predicted_solution_all_points.cpu().numpy()

    # --- Extract different physical fields from the (N, 4) flow data ---
    # The channel order is (u, v, w, p)
    
    # Pressure (p) is at channel index 3
    true_pressure = true_flow_all[:, 3]
    pred_pressure = pred_flow_all[:, 3]
    
    # Velocity magnitude is the norm of the velocity vector (u, v, w) at channels 0, 1, 2
    true_velocity_mag = np.linalg.norm(true_flow_all[:, 0:3], axis=1)
    pred_velocity_mag = np.linalg.norm(pred_flow_all[:, 0:3], axis=1)

    # 5. Visualize the results
    # ----------------------------------------------------
    print("Visualizing results...")

    # --- Visualize the Pressure Field ---
    fig_pressure = plt.figure(figsize=(14, 7))
    fig_pressure.suptitle("Pressure Field Comparison (Pa)", fontsize=16)
    # Use a consistent color range for fair comparison
    p_min = min(np.min(true_pressure), np.min(pred_pressure))
    p_max = max(np.max(true_pressure), np.max(pred_pressure))
    visualize_field(fig_pressure, 121, true_pos_all, true_pressure, "Ground Truth Pressure", v_min=p_min, v_max=p_max)
    visualize_field(fig_pressure, 122, pred_pos_all, pred_pressure, "Predicted Pressure", v_min=p_min, v_max=p_max)

    # --- Visualize the Velocity Magnitude Field ---
    fig_velocity = plt.figure(figsize=(14, 7))
    fig_velocity.suptitle("Velocity Magnitude Comparison (m/s)", fontsize=16)
    # Use a consistent color range for fair comparison
    v_min = min(np.min(true_velocity_mag), np.min(pred_velocity_mag))
    v_max = max(np.max(true_velocity_mag), np.max(pred_velocity_mag))
    visualize_field(fig_velocity, 121, true_pos_all, true_velocity_mag, "Ground Truth Velocity", cmap='viridis', v_min=v_min, v_max=v_max)
    visualize_field(fig_velocity, 122, pred_pos_all, pred_velocity_mag, "Predicted Velocity", cmap='viridis', v_min=v_min, v_max=v_max)
    
    # Display all created figures
    plt.show()


if __name__ == "__main__":
    main()
