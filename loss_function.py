# loss_function.py
import torch
from geomloss import SamplesLoss
import config

# --- Optimal Transport Loss ---
# Used to compute the reconstruction loss, measuring the distance between two point cloud distributions.
# The 'backend="auto"' setting will automatically select the fastest available backend (e.g., KeOps, PyTorch).
ot_loss_func = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, backend="auto")


def compute_ns_residuals(predicted_field, pos):
    """
    Computes the residuals of the Navier-Stokes equations to be used as a physics-informed loss (PDE Loss).
    This function must be called within a context where gradients are enabled.

    Args:
        predicted_field (torch.Tensor): The model's output tensor of shape (N, 4), representing (u, v, w, p).
        pos (torch.Tensor): The coordinates tensor of shape (N, 3), for which requires_grad must be True.

    Returns:
        torch.Tensor: A scalar tensor representing the mean squared error of the PDE residuals.
    """
    # Ensure that the position tensor can be differentiated.
    if not pos.requires_grad:
        raise ValueError("Input 'pos' must have requires_grad=True for PDE loss calculation.")

    # Physical parameters from the config file.
    rho = config.rho
    mu = config.mu

    # Decompose the predicted field into velocity components (u, v, w) and pressure (p).
    # Assumes the channel order is (u, v, w, p).
    u, v, w, p = predicted_field.split(1, dim=-1)

    # A tensor of ones is needed for the grad_outputs argument in torch.autograd.grad.
    ones = torch.ones_like(p)
    
    # --- First-order derivatives ---
    grad_p = torch.autograd.grad(p, pos, grad_outputs=ones, create_graph=True)[0]
    dp_dx, dp_dy, dp_dz = grad_p.split(1, dim=-1)

    grad_u = torch.autograd.grad(u, pos, grad_outputs=ones, create_graph=True)[0]
    du_dx, du_dy, du_dz = grad_u.split(1, dim=-1)

    grad_v = torch.autograd.grad(v, pos, grad_outputs=ones, create_graph=True)[0]
    dv_dx, dv_dy, dv_dz = grad_v.split(1, dim=-1)

    grad_w = torch.autograd.grad(w, pos, grad_outputs=ones, create_graph=True)[0]
    dw_dx, dw_dy, dw_dz = grad_w.split(1, dim=-1)
    
    # --- Second-order derivatives (Laplacian operator) ---
    # Calculated by taking the gradient of the first-order derivatives.
    laplacian_u = torch.autograd.grad(du_dx, pos, grad_outputs=ones, create_graph=True)[0][:, 0:1] + \
                  torch.autograd.grad(du_dy, pos, grad_outputs=ones, create_graph=True)[0][:, 1:2] + \
                  torch.autograd.grad(du_dz, pos, grad_outputs=ones, create_graph=True)[0][:, 2:3]

    laplacian_v = torch.autograd.grad(dv_dx, pos, grad_outputs=ones, create_graph=True)[0][:, 0:1] + \
                  torch.autograd.grad(dv_dy, pos, grad_outputs=ones, create_graph=True)[0][:, 1:2] + \
                  torch.autograd.grad(dv_dz, pos, grad_outputs=ones, create_graph=True)[0][:, 2:3]

    laplacian_w = torch.autograd.grad(dw_dx, pos, grad_outputs=ones, create_graph=True)[0][:, 0:1] + \
                  torch.autograd.grad(dw_dy, pos, grad_outputs=ones, create_graph=True)[0][:, 1:2] + \
                  torch.autograd.grad(dw_dz, pos, grad_outputs=ones, create_graph=True)[0][:, 2:3]

    # --- PDE Residuals ---
    # Continuity equation (conservation of mass for incompressible flow)
    continuity_residual = du_dx + dv_dy + dw_dz

    # Momentum equations (steady-state, ignoring gravity)
    # x-momentum
    momentum_x = rho * (u * du_dx + v * du_dy + w * du_dz) + dp_dx - mu * laplacian_u
    # y-momentum
    momentum_y = rho * (u * dv_dx + v * dv_dy + w * dv_dz) + dp_dy - mu * laplacian_v
    # z-momentum
    momentum_z = rho * (u * dw_dx + v * dw_dy + w * dw_dz) + dp_dz - mu * laplacian_w

    # Calculate the mean squared error of the residuals to form the loss.
    loss_continuity = torch.mean(continuity_residual ** 2)
    loss_momentum = torch.mean(momentum_x ** 2) + torch.mean(momentum_y ** 2) + torch.mean(momentum_z ** 2)

    return loss_continuity + loss_momentum


def cvae_boundary_driven_loss(predicted_solution, true_solution, mu, log_var, query_pos_batch, query_pos, compute_pde=True):
    """
    Computes the total CVAE loss, which is a weighted sum of reconstruction, KLD, and PDE losses.
    Total Loss = w1 * OT_Loss + w2 * KLD_Loss + w3 * PDE_Loss

    Args:
        predicted_solution (torch.Tensor): The reconstructed solution from the decoder.
        true_solution (torch.Tensor): The ground truth solution.
        mu (torch.Tensor): The mean of the latent space distribution.
        log_var (torch.Tensor): The log variance of the latent space distribution.
        query_pos_batch (torch.Tensor): The batch index for each query point.
        query_pos (torch.Tensor): The coordinates of the query points.
        compute_pde (bool): Flag to enable/disable PDE loss calculation. Can be set to False to speed up validation.

    Returns:
        tuple: A tuple containing (total_loss, ot_loss, kld_loss, pde_loss).
    """
    # --- 1. OT Loss (Reconstruction Loss) ---
    ot_loss = 0.0
    # OT loss must be computed per sample in the batch, as each sample is an independent distribution.
    unique_batches = torch.unique(query_pos_batch)
    for batch_idx in unique_batches:
        mask = (query_pos_batch == batch_idx)
        if mask.sum() > 0:  # Ensure there are points for this sample
            ot_loss += ot_loss_func(predicted_solution[mask], true_solution[mask])
    # Average the OT loss over the number of samples in the batch.
    ot_loss /= len(unique_batches)

    # --- 2. KL-Divergence Loss ---
    # This measures the difference between the posterior distribution q(z|y,c) and the prior p(z).
    # 'sum' is performed over the feature dimension, and 'mean' is over the batch dimension.
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kld_loss = torch.mean(kld_loss)

    # --- 3. PDE Loss (Physics-Informed Loss) ---
    if compute_pde:
        pde_loss = compute_ns_residuals(predicted_solution, query_pos)
    else:
        # If not computed, return a zero tensor to maintain a consistent return structure.
        pde_loss = torch.tensor(0.0, device=predicted_solution.device)
        
    # --- 4. Total Weighted Loss ---
    # Combine the individual losses using weights from the config file.
    total_loss = (config.LAMBDA_OT * ot_loss +
                  config.BETA_VAE * kld_loss +
                  config.LAMBDA_PDE * pde_loss)

    return total_loss, ot_loss, kld_loss, pde_loss
