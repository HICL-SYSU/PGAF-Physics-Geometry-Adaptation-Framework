# predict.py
import torch
import numpy as np
import matplotlib
# Use a non-interactive backend for server environments like Code Ocean
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse  # For command-line arguments

# Import project-specific modules
import config
from model import PGAF
from mydata import MultiFolderDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def visualize_and_save_field(pos, true_field, pred_field, title, filename, save_dir, cmap='jet'):
    """
    A helper function to create, display, and save a comparison plot for a scalar field.
    """
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"{title} Comparison", fontsize=16)

    # Use a consistent color range for fair comparison
    v_min = min(np.min(true_field), np.min(pred_field))
    v_max = max(np.max(true_field), np.max(pred_field))

    # --- Plot Ground Truth ---
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=true_field, cmap=cmap, s=5, vmin=v_min, vmax=v_max)
    ax1.set_title("Ground Truth")
    ax1.view_init(elev=30, azim=45)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    fig.colorbar(sc1, ax=ax1, shrink=0.6)

    # --- Plot Prediction ---
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=pred_field, cmap=cmap, s=5, vmin=v_min, vmax=v_max)
    ax2.set_title("Prediction")
    ax2.view_init(elev=30, azim=45)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    fig.colorbar(sc2, ax=ax2, shrink=0.6)

    # --- Save the figure ---
    # Construct a descriptive filename
    save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_{title.lower().replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Close the figure to free up memory, which is crucial in a long loop
    plt.close(fig)


def main():
    """
    Main execution function for prediction and visualization on the entire test set.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Run prediction and visualization for the PGAF model.")
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=None, 
        help="Path to the trained model (.pth file). If not provided, uses the path from config.py."
    )
    args = parser.parse_args()

    print(f"--- Final Model Evaluation on the Test Set ---")
    
    # Determine which model path to use: from command line or from config file
    model_load_path = args.model_path if args.model_path else config.MODEL_SAVE_PATH
    
    # Define and create the save directory for pictures
    pic_save_dir = "/results/pic"
    os.makedirs(pic_save_dir, exist_ok=True)
    print(f"Plots will be saved to: {pic_save_dir}")

    device = config.DEVICE

    # 1. Load the test dataset and the trained model
    # ----------------------------------------------------
    print(f"Loading best model from: {model_load_path}")
    test_dataset = MultiFolderDataset(root_dir=config.DATA_DIR, split='test')
    if len(test_dataset) == 0:
        print("Test dataset is empty. Cannot perform final evaluation.")
        return
    
    # Use follow_batch to ensure `query_pos_batch` is created correctly by the DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, follow_batch=['query_pos'])

    model = PGAF().to(device)
    if not os.path.exists(model_load_path):
        print(f"Model file not found at {model_load_path}. Please train the model first.")
        return
    
    # Load the model state dictionary, using weights_only=True for security
    model.load_state_dict(torch.load(model_load_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Loop through all samples in the test set
    # ----------------------------------------------------
    print(f"\nProcessing {len(test_loader)} samples from the test set...")
    # [FIXED] Use enumerate to get an index 'i' for each batch to reliably get metadata.
    for i, data_sample in enumerate(tqdm(test_loader, desc="Predicting")):
        
        # [FIXED] Get the filename safely from the original dataset using the index.
        # This is the most robust way, as DataLoader might not preserve non-tensor attributes.
        sample_filename = test_dataset.sample_files[i]
        
        # --- Prepare a new, clean Data object for inference ---
        all_query_pos = torch.cat([data_sample.pos, data_sample.query_pos], dim=0)
        
        inference_data = Data(
            x=data_sample.x,
            pos=data_sample.pos,
            query_pos=all_query_pos
        )
        # Manually create the necessary batch index attributes for the new Data object.
        inference_data.batch = torch.zeros(data_sample.pos.size(0), dtype=torch.long)
        inference_data.query_pos_batch = torch.zeros(all_query_pos.size(0), dtype=torch.long)
        
        inference_data = inference_data.to(device)

        # --- Run Inference ---
        with torch.no_grad():
            predicted_solution_all_points = model(inference_data)

        # --- Prepare data for plotting ---
        true_pos_all = torch.cat([data_sample.pos, data_sample.query_pos], dim=0).cpu().numpy()
        true_flow_all = torch.cat([data_sample.x, data_sample.y], dim=0).cpu().numpy()
        pred_pos_all = all_query_pos.cpu().numpy()
        pred_flow_all = predicted_solution_all_points.cpu().numpy()

        # --- Extract physical fields ---
        # Channel order is (u, v, w, p), so pressure 'p' is at index 3.
        true_pressure = true_flow_all[:, 3]
        pred_pressure = pred_flow_all[:, 3]
        # Velocity magnitude is the norm of the velocity vector (u, v, w) at channels 0, 1, 2.
        true_velocity_mag = np.linalg.norm(true_flow_all[:, 0:3], axis=1)
        pred_velocity_mag = np.linalg.norm(pred_flow_all[:, 0:3], axis=1)
        
        # --- Visualize and Save ---
        # visualize_and_save_field(
        #     true_pos_all, true_pressure, pred_pressure, 
        #     title="Pressure Field", filename=sample_filename, save_dir=pic_save_dir, cmap='jet'
        # )
        visualize_and_save_field(
            true_pos_all, true_velocity_mag, pred_velocity_mag, 
            title="Velocity Magnitude", filename=sample_filename, save_dir=pic_save_dir, cmap='viridis'
        )

    print(f"\nPrediction and visualization complete. Check the '{pic_save_dir}' directory.")


if __name__ == "__main__":
    main()
