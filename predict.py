# predict.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

import config
from model import PGAF
from mydata import MultiFolderDataset
from torch_geometric.data import DataLoader


def main():
    print(f"--- Final Model Evaluation on the Test Set ---")
    print(f"Loading best model from {config.MODEL_SAVE_PATH}")
    device = config.DEVICE

    test_dataset = MultiFolderDataset(root_dir=config.DATA_DIR, split='test')
    if len(test_dataset) == 0:
        print("Test dataset is empty. Cannot perform final evaluation. Please populate the 'test' directory.")
        return

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, follow_batch=['query_pos'])

    data_sample = next(iter(test_loader)).to(device)

    model = PGAF().to(device)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Model file not found at {config.MODEL_SAVE_PATH}. Please train the model first.")
        return
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # --- Inference ---
    with torch.no_grad():
        generated_solution_1 = model(data_sample)
        generated_solution_2 = model(data_sample)

    print("Prediction on test sample complete. Visualizing results...")

    # --- Visualization ---
    boundary_pos = data_sample.pos.cpu().numpy()
    query_pos = data_sample.query_pos.cpu().numpy()
    true_solution_query = data_sample.y.cpu().numpy()
    pred_solution_1_query = generated_solution_1.cpu().numpy()
    boundary_flow = data_sample.x.cpu().numpy()

    all_pos_true = np.vstack([boundary_pos, query_pos])
    all_flow_true = np.vstack([boundary_flow, true_solution_query])
    all_pos_pred1 = np.vstack([boundary_pos, query_pos])
    all_flow_pred1 = np.vstack([boundary_flow, pred_solution_1_query])

    true_p = all_flow_true[:, 0]
    pred1_p = all_flow_pred1[:, 0]

    fig = plt.figure(figsize=(12, 6))
    v_min, v_max = np.min(true_p), np.max(true_p)

    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(all_pos_true[:, 0], all_pos_true[:, 1], all_pos_true[:, 2], c=true_p, cmap='jet', s=5, vmin=v_min,
                      vmax=v_max)
    ax1.set_title("Ground Truth Pressure")
    fig.colorbar(sc1, ax=ax1, shrink=0.6)

    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(all_pos_pred1[:, 0], all_pos_pred1[:, 1], all_pos_pred1[:, 2], c=pred1_p, cmap='jet', s=5,
                      vmin=v_min, vmax=v_max)
    ax2.set_title("Predicted Pressure")
    fig.colorbar(sc2, ax=ax2, shrink=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()