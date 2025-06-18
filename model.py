# model.py (Deeper and Optimized CVAE Boundary-Driven Version)
import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, global_max_pool, fps, knn
import config


class ResMLP(nn.Module):
    """A multilayer perceptron block with residual connections"""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),
        )
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.mlp(x) + self.shortcut(x)


class SA_Module(nn.Module):
    """Encapsulated Set Abstraction Module (Encoder Block)"""

    def __init__(self, ratio, k, in_channels, mlp_channels):
        super().__init__()
        self.ratio = ratio
        self.k = k
        self.conv = PointNetConv(
            nn.Sequential(
                ResMLP(in_channels + 3, mlp_channels[0], mlp_channels[1]),
                nn.ReLU(inplace=True)
            )
        )

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = knn(pos, pos[idx], k=self.k, batch_x=batch, batch_y=batch[idx])
        edge_index = torch.stack([col, row], dim=0)

        x_dest = x[idx] if x is not None else None
        x_out = self.conv((x, x_dest), (pos, pos[idx]), edge_index)

        pos_out, batch_out = pos[idx], batch[idx]
        return x_out, pos_out, batch_out


class FP_Module(nn.Module):
    """Encapsulated Feature Propagation Module (Decoder Block)"""

    def __init__(self, k, mlp_channels):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            ResMLP(mlp_channels[0], mlp_channels[1], mlp_channels[2]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_target, pos_target, batch_target, x_source, pos_source, batch_source):
        interpolated = knn_interpolate(
            x_source, pos_source, pos_target, batch_source, batch_target, k=self.k
        )
        if x_target is not None:
            combined = torch.cat([x_target, interpolated], dim=1)
        else:
            combined = interpolated

        return self.mlp(combined)


# --- main model ---
class ConditionEncoder(nn.Module):
    """
Encoder: Extract global condition vector c from boundary conditions
    """

    def __init__(self, in_channels, condition_dim):
        super().__init__()

        self.sa1 = SA_Module(0.5, 16, in_channels, [in_channels + 3, 64])
        self.sa2 = SA_Module(0.5, 16, 64, [64 + 3, 128])
        self.sa3 = SA_Module(0.5, 16, 128, [128 + 3, 256])
        self.sa4 = SA_Module(0.5, 16, 256, [256 + 3, 512])

        # Global feature processing
        self.global_mlp = nn.Sequential(
            ResMLP(512, 512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, condition_dim)
        )

    def forward(self, x, pos, batch):
        x1, pos1, batch1 = self.sa1(x, pos, batch)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)
        x3, pos3, batch3 = self.sa3(x2, pos2, batch2)
        x4, _, _ = self.sa4(x3, pos3, batch3)

        global_feat = global_max_pool(x4, batch3)
        return self.global_mlp(global_feat)


class PosteriorEncoder(nn.Module):
    """
(Used during training) Infer the posterior distribution of z based on the internal true solution y and condition c
    """

    def __init__(self, condition_dim, latent_dim):
        super().__init__()
        # Encode the internal solution
        self.in_channels = config.FIELD_CHANNELS
        self.sa1 = SA_Module(0.25, 16, self.in_channels, [self.in_channels + 3, 64])
        self.sa2 = SA_Module(0.25, 16, 64, [64 + 3, 128])
        self.sa3 = SA_Module(0.25, 16, 128, [128 + 3, 256])

        # Concatenate the conditional vector c and process
        self.final_mlp = nn.Sequential(
            ResMLP(256 + condition_dim, 512, 512),
            nn.ReLU(inplace=True)
        )
        self.mu_layer = nn.Linear(512, latent_dim)
        self.log_var_layer = nn.Linear(512, latent_dim)

    def forward(self, y, query_pos, query_pos_batch, c):
        x1, pos1, batch1 = self.sa1(y, query_pos, query_pos_batch)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)
        x3, _, _ = self.sa3(x2, pos2, batch2)

        y_feat = global_max_pool(x3, batch2)
        combined_feat = torch.cat([y_feat, c], dim=1)

        h = self.final_mlp(combined_feat)
        return self.mu_layer(h), self.log_var_layer(h)


class InteriorDecoder(nn.Module):
    """
Decoder: receives z, c and internal point coordinates and reconstructs the solution
    """

    def __init__(self, latent_dim, condition_dim, out_channels):
        super().__init__()
        # Encode the geometric features of the query point
        self.sa1 = SA_Module(0.25, 16, None, [3, 64])
        self.sa2 = SA_Module(0.25, 16, 64, [64 + 3, 128])
        self.sa3 = SA_Module(0.25, 16, 128, [128 + 3, 256])

        self.bottleneck_mlp = nn.Sequential(
            ResMLP(256 + latent_dim + condition_dim, 512, 512),
            nn.ReLU(inplace=True)
        )

        # Feature Propagation Module
        self.fp1 = FP_Module(3, [512 + 128, 256, 256])
        self.fp2 = FP_Module(3, [256 + 64, 128, 128])
        self.fp3 = FP_Module(3, [128, 128, 128])

        # output
        self.final_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, out_channels)
        )

    def forward(self, z, c, query_pos, query_pos_batch):
        # Encode the geometry of the query point
        x1, pos1, batch1 = self.sa1(None, query_pos, query_pos_batch)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)
        x3, _, _ = self.sa3(x2, pos2, batch2)

        z_expanded = z[batch2]
        c_expanded = c[batch2]
        bottleneck_feat = torch.cat([x3, z_expanded, c_expanded], dim=1)
        bottleneck_feat = self.bottleneck_mlp(bottleneck_feat)

        # Decode
        up_x2 = self.fp1(x2, pos2, batch2, bottleneck_feat, pos2, batch2)
        up_x1 = self.fp2(x1, pos1, batch1, up_x2, pos2, batch2)
        final_x = self.fp3(None, query_pos, query_pos_batch, up_x1, pos1, batch1)

        return self.final_mlp(final_x)


class PGAF(nn.Module):
    """
    Main model: Combined deepened encoder and decoder
    """

    def __init__(self):
        super().__init__()
        condition_dim = 256

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
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        boundary_x, boundary_pos, boundary_batch = data.x, data.pos, data.batch
        query_pos, query_pos_batch = data.query_pos, data.query_pos_batch

        c = self.condition_encoder(boundary_x, boundary_pos, boundary_batch)

        if self.training and hasattr(data, 'y'):
            true_solution = data.y
            mu, log_var = self.posterior_encoder(true_solution, query_pos, query_pos_batch, c)
            z = self.reparameterize(mu, log_var)
            reconstructed_solution = self.decoder(z, c, query_pos, query_pos_batch)
            return reconstructed_solution, mu, log_var
        else:
            batch_size = c.size(0)
            z = torch.randn(batch_size, config.LATENT_DIM).to(c.device)
            generated_solution = self.decoder(z, c, query_pos, query_pos_batch)
            return generated_solution