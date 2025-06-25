# mydata.py (Modified to include point cloud sampling)
import os
import torch
import numpy as np
from torch_geometric.data import Dataset, Data, DataLoader
from torch.utils.data import random_split

# Import the config file to get sampling parameters
import config


def sample_points(points, features, num_samples):
    """
    A helper function to randomly sample a fixed number of points and their corresponding features.

    Args:
        points (torch.Tensor): The coordinates tensor, shape (N, 3).
        features (torch.Tensor): The feature tensor, shape (N, D).
        num_samples (int): The desired number of points to sample.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The sampled points and features.
    """
    num_total_points = points.size(0)
    
    # If the number of points is less than or equal to the desired sample size,
    # we might need to sample with replacement or handle it differently.
    # For simplicity and robustness, we will sample with replacement if needed.
    # This prevents errors when a sample has fewer points than num_samples.
    replace = num_total_points < num_samples
    if replace:
        print(f"Warning: Not enough points ({num_total_points}) to sample {num_samples} without replacement. Sampling with replacement.")

    # Generate random indices.
    # We use torch.randperm for non-replacement sampling and torch.randint for replacement sampling.
    if not replace:
        indices = torch.randperm(num_total_points)[:num_samples]
    else:
        indices = torch.randint(0, num_total_points, (num_samples,))

    # Use the indices to select the sampled points and features.
    sampled_points = points[indices]
    sampled_features = features[indices]

    return sampled_points, sampled_features


class MultiFolderDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.data_path = os.path.join(self.root_dir, self.split)

        # Folder Names
        self.in_out_coords_dir = os.path.join(self.data_path, 'Inlet&Outlet_coords')
        self.in_out_flow_dir = os.path.join(self.data_path, 'Inlet&Outlet_flow')
        self.inter_coords_dir = os.path.join(self.data_path, 'Inter_coords')
        self.inter_flow_dir = os.path.join(self.data_path, 'Inter_flow')
        self.wall_coords_dir = os.path.join(self.data_path, 'Wall_coords')
        self.wall_flow_dir = os.path.join(self.data_path, 'Wall_flow')

        if os.path.exists(self.in_out_coords_dir):
            self.sample_files = sorted(os.listdir(self.in_out_coords_dir))
        else:
            self.sample_files = []

        super().__init__(root_dir)

    def len(self):
        return len(self.sample_files)

    def get(self, idx):
        sample_filename = self.sample_files[idx]

        # --- Load all data from disk ---
        # Note: We load them as torch tensors directly.
        in_out_coords = torch.from_numpy(np.load(os.path.join(self.in_out_coords_dir, sample_filename)))
        in_out_flow = torch.from_numpy(np.load(os.path.join(self.in_out_flow_dir, sample_filename)))
        inter_coords = torch.from_numpy(np.load(os.path.join(self.inter_coords_dir, sample_filename)))
        wall_coords = torch.from_numpy(np.load(os.path.join(self.wall_coords_dir, sample_filename)))
        inter_flow_solution = torch.from_numpy(np.load(os.path.join(self.inter_flow_dir, sample_filename)))
        wall_flow_solution = torch.from_numpy(np.load(os.path.join(self.wall_flow_dir, sample_filename)))

        # --- Combine interior and wall points to form the full query set ---
        full_query_pos = torch.cat([inter_coords, wall_coords], dim=0)
        full_ground_truth = torch.cat([inter_flow_solution, wall_flow_solution], dim=0)

        # --- [MODIFICATION] Perform sampling ---
        
        # Sample boundary points (inlet/outlet)
        if config.NUM_BOUNDARY_POINTS is not None:
            sampled_boundary_pos, sampled_boundary_flow = sample_points(
                in_out_coords, in_out_flow, config.NUM_BOUNDARY_POINTS
            )
        else:
            # If NUM_BOUNDARY_POINTS is None, use all points.
            sampled_boundary_pos = in_out_coords
            sampled_boundary_flow = in_out_flow

        # Sample query points (interior/wall)
        if config.NUM_QUERY_POINTS is not None:
            sampled_query_pos, sampled_ground_truth = sample_points(
                full_query_pos, full_ground_truth, config.NUM_QUERY_POINTS
            )
        else:
            # If NUM_QUERY_POINTS is None, use all points.
            sampled_query_pos = full_query_pos
            sampled_ground_truth = full_ground_truth
            
        # --- Create a PyG Data Object with the SAMPLED data ---
        data = Data(
            x=sampled_boundary_flow.float(),      # Features for condition points
            pos=sampled_boundary_pos.float(),     # Positions for condition points
            query_pos=sampled_query_pos.float(),  # Positions for query points
            y=sampled_ground_truth.float()        # Ground truth at query points
        )
        return data


# --- The rest of the file remains unchanged ---
def get_dataloaders_with_split(root_dir, batch_size, val_split_ratio=0.2, num_workers=0):
    # This function does not need any changes, as the sampling is handled inside the Dataset class.
    # ... (rest of the function is identical to your original)
    # 1. Load all the data in the complete 'train' folder
    full_train_dataset = MultiFolderDataset(root_dir=root_dir, split='train')

    if len(full_train_dataset) == 0:
        raise FileNotFoundError("No data found in the 'train' directory. Please check your data path.")

    # 2.Calculate the size of the training and validation sets
    num_total_samples = len(full_train_dataset)
    val_size = int(val_split_ratio * num_total_samples)
    train_size = num_total_samples - val_size

    print(f"Total training samples: {num_total_samples}")
    print(f"Splitting into {train_size} training samples and {val_size} validation samples.")

    # 3. Pinning seeds to ensure reproducibility
    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 4. DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        follow_batch=['query_pos']
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        follow_batch=['query_pos']
    )

    # 5. test set
    test_dataset = MultiFolderDataset(root_dir=root_dir, split='test')
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            follow_batch=['query_pos']
        )
    else:
        print("Warning: No data found in the 'test' directory. The test loader will be empty.")
        test_loader = None

    return train_loader, val_loader, test_loader
