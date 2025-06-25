# model.py (Corrected version with English comments)
import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, global_max_pool, fps, knn, knn_interpolate
import config


# --- Helper Modules ---

class ResMLP(nn.Module):
    """A Multi-Layer Perceptron (MLP) block with a residual connection."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),
        )
        # The shortcut connection handles cases where input and output dimensions differ.
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.mlp(x) + self.shortcut(x)


class SA_Module(nn.Module):
    """
    An encapsulated Set Abstraction (SA) module, a core component of PointNet++.
    This module performs downsampling and feature extraction on a point cloud.
    """
    def __init__(self, ratio, k, in_channels, mlp_channels):
        super().__init__()
        self.ratio = ratio  # The ratio of points to sample (downsampling).
        self.k = k          # The number of nearest neighbors to consider for each sampled point.
        
        # Determine the input dimension for the PointNetConv's MLP.
        # If in_channels is None, only coordinate information is used (3 dims).
        # Otherwise, it's the sum of feature dimension and coordinate dimension (3).
        mlp_in_dim = 3 if in_channels is None else in_channels + 3
        
        self.conv = PointNetConv(
            nn.Sequential(
                # The ResMLP's input is mlp_in_dim, with hidden and output layers defined by mlp_channels.
                ResMLP(mlp_in_dim, mlp_channels[0], mlp_channels[1]),
                nn.ReLU(inplace=True)
            )
        )

    def forward(self, x, pos, batch):
        # 1. Downsample the points using Farthest Point Sampling (FPS).
        idx = fps(pos, batch, ratio=self.ratio)
        
        # 2. Group neighboring points for each sampled point using K-Nearest Neighbors (KNN).
        row, col = knn(pos, pos[idx], self.k, batch_x=batch, batch_y=batch[idx])
        edge_index = torch.stack([col, row], dim=0)

        # 3. Apply the PointNet-like convolution to extract features.
        x_dest = x[idx] if x is not None else None
        x_out = self.conv((x, x_dest), (pos, pos[idx]), edge_index)

        # Return the features, positions, and batch indices of the downsampled point cloud.
        pos_out, batch_out = pos[idx], batch[idx]
        return x_out, pos_out, batch_out


class FP_Module(nn.Module):
    """
    An encapsulated Feature Propagation (FP) module, another core component of PointNet++.
    This module interpolates features from a sparse point cloud to a denser one.
    """
    def __init__(self, k, mlp_channels):
        super().__init__()
        self.k = k
        # The mlp_channels for FP modules are typically [in_channels, hidden_channels, out_channels].
        self.mlp = nn.Sequential(
            ResMLP(mlp_channels[0], mlp_channels[1], mlp_channels[2]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_target, pos_target, batch_target, x_source, pos_source, batch_source):
        # 1. Interpolate features from the source point cloud to the target points using KNN.
        interpolated = knn_interpolate(
            x_source, pos_source, pos_target, batch_x=batch_source, batch_y=batch_target, k=self.k
        )
        
        # 2. Concatenate the interpolated features with the original features of the target points (if they exist).
        if x_target is not None:
            combined = torch.cat([x_target, interpolated], dim=1)
        else:
            combined = interpolated
        
        # 3. Pass the combined features through an MLP to fuse them.
        return self.mlp(combined)


# --- Main CVAE Model Components ---

class ConditionEncoder(nn.Module):
    """Encoder: Extracts a global condition vector 'c' from the boundary conditions (BC)."""
    def __init__(self, in_channels, condition_dim):
        super().__init__()
        
        # A hierarchical feature extractor based on stacked SA modules (PointNet++ encoder).
        # Architecture: input_features -> 64 -> 128 -> 256 -> 512
        self.sa1 = SA_Module(0.5, 16, in_channels, mlp_channels=[32, 64])
        self.sa2 = SA_Module(0.5, 16, 64, mlp_channels=[64, 128])
        self.sa3 = SA_Module(0.5, 16, 128, mlp_channels=[128, 256])
        self.sa4 = SA_Module(0.5, 16, 256, mlp_channels=[256, 512])

        # An MLP to process the global feature vector after max pooling.
        self.global_mlp = nn.Sequential(
            ResMLP(512, 512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, condition_dim)
        )

    def forward(self, x, pos, batch):
        # Pass the BC point cloud through the SA layers.
        x1, pos1, batch1 = self.sa1(x, pos, batch)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)
        x3, pos3, batch3 = self.sa3(x2, pos2, batch2)
        x4, pos4, batch4 = self.sa4(x3, pos3, batch3)

        # Aggregate features into a single global vector per sample.
        global_feat = global_max_pool(x4, batch4)
        return self.global_mlp(global_feat)


class PosteriorEncoder(nn.Module):
    """(Used only during training) Infers the posterior distribution of z, q(z|y,c), from the ground truth solution 'y' and condition 'c'."""
    def __init__(self, condition_dim, latent_dim):
        super().__init__()
        self.in_channels = config.FIELD_CHANNELS
        
        # Another PointNet++ encoder to process the ground truth solution field 'y'.
        # Architecture: input_features -> 64 -> 128 -> 256
        self.sa1 = SA_Module(0.25, 16, self.in_channels, mlp_channels=[32, 64])
        self.sa2 = SA_Module(0.25, 16, 64, mlp_channels=[64, 128])
        self.sa3 = SA_Module(0.25, 16, 128, mlp_channels=[128, 256])

        # MLP to combine the feature of 'y' and the condition 'c'.
        self.final_mlp = nn.Sequential(
            ResMLP(256 + condition_dim, 512, 512),
            nn.ReLU(inplace=True)
        )
        # Output layers for the mean (mu) and log-variance (log_var) of the latent distribution.
        self.mu_layer = nn.Linear(512, latent_dim)
        self.log_var_layer = nn.Linear(512, latent_dim)

    def forward(self, y, query_pos, query_pos_batch, c):
        # Encode the ground truth solution 'y'.
        x1, pos1, batch1 = self.sa1(y, query_pos, query_pos_batch)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)
        x3, pos3, batch3 = self.sa3(x2, pos2, batch2)

        y_feat = global_max_pool(x3, batch3)
        # Combine the feature from 'y' with the condition vector 'c'.
        combined_feat = torch.cat([y_feat, c], dim=1)

        # Output mu and log_var.
        h = self.final_mlp(combined_feat)
        return self.mu_layer(h), self.log_var_layer(h)


class InteriorDecoder(nn.Module):
    """Decoder: Reconstructs the solution field given the latent vector 'z', condition 'c', and query point coordinates."""
    def __init__(self, latent_dim, condition_dim, out_channels):
        super().__init__()
        # PointNet++ encoder part to extract multi-scale geometric features from the query points' coordinates.
        # Note: The first SA module has `in_channels=None` as it only processes coordinates.
        self.sa1 = SA_Module(0.25, 16, None, mlp_channels=[32, 64]) # Encodes 3D coordinates to 64D features.
        self.sa2 = SA_Module(0.25, 16, 64, mlp_channels=[64, 128])
        self.sa3 = SA_Module(0.25, 16, 128, mlp_channels=[128, 256])

        # MLP for the bottleneck layer, where geometric features, z, and c are fused.
        self.bottleneck_mlp = nn.Sequential(
            ResMLP(256 + latent_dim + condition_dim, 512, 512),
            nn.ReLU(inplace=True)
        )

        # Feature Propagation (FP) modules for upsampling and decoding.
        # The input dimension is the sum of the target features and the interpolated source features.
        self.fp1 = FP_Module(3, [512 + 256, 256, 256]) # from bottleneck(512) to sa3(256)
        self.fp2 = FP_Module(3, [256 + 128, 128, 128]) # from fp1_out(256) to sa2(128)
        self.fp3 = FP_Module(3, [128 + 64, 128, 128])  # from fp2_out(128) to sa1(64)
        self.fp4 = FP_Module(3, [128, 128, 128])       # from fp3_out(128) to original points (no features)

        # Final output layer to produce the predicted flow field.
        self.final_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, out_channels)
        )

    def forward(self, z, c, query_pos, query_pos_batch):
        # 1. Encode the geometric structure of the query points.
        x1_geom, pos1, batch1 = self.sa1(None, query_pos, query_pos_batch)
        x2_geom, pos2, batch2 = self.sa2(x1_geom, pos1, batch1)
        x3_geom, pos3, batch3 = self.sa3(x2_geom, pos2, batch2)

        # 2. Expand z and c to match the number of points in the bottleneck layer and fuse them.
        # This is a general way to broadcast batch-level vectors (z, c) to point-level features using PyG's batch index.
        z_expanded = z[batch3]
        c_expanded = c[batch3]

        bottleneck_feat = torch.cat([x3_geom, z_expanded, c_expanded], dim=1)
        bottleneck_feat = self.bottleneck_mlp(bottleneck_feat)

        # 3. Decode by propagating features from coarse to fine levels.
        up_x3 = self.fp1(x3_geom, pos3, batch3, bottleneck_feat, pos3, batch3)
        up_x2 = self.fp2(x2_geom, pos2, batch2, up_x3, pos3, batch3)
        up_x1 = self.fp3(x1_geom, pos1, batch1, up_x2, pos2, batch2)
        final_x = self.fp4(None, query_pos, query_pos_batch, up_x1, pos1, batch1)

        # 4. Generate the final prediction.
        return self.final_mlp(final_x)


# --- The Main Model ---

class PGAF(nn.Module):
    """
    Physics-Guided Autoregressive Flow (or similar name).
    This is the main CVAE model that combines all the components.
    """
    def __init__(self):
        super().__init__()
        condition_dim = 256 # Dimension of the global condition vector.

        self.condition_encoder = ConditionEncoder(
            in_channels=config.FIELD_CHANNELS,
            condition_dim=condition_dim
        )
        self.posterior_encoder = PosteriorEncoder(
            condition_dim=condition_dim,
            latent_dim=config.LATENT_DIM
        )
        self.decoder = InteriorDecoder(
            latent_dim=config.LATENT_DIM,
            condition_dim=condition_dim,
            out_channels=config.FIELD_CHANNELS
        )

    def reparameterize(self, mu, log_var):
        """The reparameterization trick to allow backpropagation through a random node."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        # Unpack data from the PyG Data object.
        boundary_x, boundary_pos, boundary_batch = data.x, data.pos, data.batch
        query_pos, query_pos_batch = data.query_pos, data.query_pos_batch

        # 1. Encode the boundary conditions to get the condition vector 'c'.
        c = self.condition_encoder(boundary_x, boundary_pos, boundary_batch)

        # 2. Follow different paths for training and inference.
        if self.training and hasattr(data, 'y'):
            # --- Training Path ---
            # Use the ground truth solution 'y' to generate the posterior distribution.
            true_solution = data.y
            mu, log_var = self.posterior_encoder(true_solution, query_pos, query_pos_batch, c)
            # Sample from the posterior distribution using the reparameterization trick.
            z = self.reparameterize(mu, log_var)
            # Decode to reconstruct the solution.
            reconstructed_solution = self.decoder(z, c, query_pos, query_pos_batch)
            return reconstructed_solution, mu, log_var
        else:
            # --- Inference/Generation Path ---
            # Sample 'z' from the prior distribution (a standard normal distribution).
            batch_size = c.size(0)
            z = torch.randn(batch_size, config.LATENT_DIM, device=c.device)
            # Decode to generate a new solution.
            generated_solution = self.decoder(z, c, query_pos, query_pos_batch)
            return generated_solution
