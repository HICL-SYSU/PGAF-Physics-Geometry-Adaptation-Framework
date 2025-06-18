# mydata.py (完整，包含动态划分功能)
import os
import torch
import numpy as np
from glob import glob
from torch_geometric.data import Dataset, Data, DataLoader
from torch.utils.data import random_split


class MultiFolderDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.data_path = os.path.join(self.root_dir, self.split)

        # Folder Name
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

        # --- load data ---
        in_out_coords = torch.from_numpy(np.load(os.path.join(self.in_out_coords_dir, sample_filename)))
        in_out_flow = torch.from_numpy(np.load(os.path.join(self.in_out_flow_dir, sample_filename)))
        inter_coords = torch.from_numpy(np.load(os.path.join(self.inter_coords_dir, sample_filename)))
        wall_coords = torch.from_numpy(np.load(os.path.join(self.wall_coords_dir, sample_filename)))
        inter_flow_solution = torch.from_numpy(np.load(os.path.join(self.inter_flow_dir, sample_filename)))
        wall_flow_solution = torch.from_numpy(np.load(os.path.join(self.wall_flow_dir, sample_filename)))

        query_points_pos = torch.cat([inter_coords, wall_coords], dim=0)
        ground_truth_solution = torch.cat([inter_flow_solution, wall_flow_solution], dim=0)

        # --- Creating a PyG Data Object ---
        data = Data(
            x=in_out_flow.float(),
            pos=in_out_coords.float(),
            query_pos=query_points_pos.float(),
            y=ground_truth_solution.float()
        )
        return data


# --- Divide the dataset ---
def get_dataloaders_with_split(root_dir, batch_size, val_split_ratio=0.2, num_workers=0):

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