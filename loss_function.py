# loss_function.py
import torch
from geomloss import SamplesLoss
import config

# --- Optimal Transport Loss ---
# Used to calculate the reconstruction loss, measuring the distance between the two point cloud distributionsot_loss_func = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, backend="auto")
ot_loss_func = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, backend="auto")

def compute_ns_residuals(predicted_field, pos):
    """
    Navier-Stokes Loss
    """
    # Blood Properties
    rho = config.rho
    mu = config.mu

    # split velocity and pressure
    u = predicted_field[:, 0:1]
    v = predicted_field[:, 1:2]
    w = predicted_field[:, 2:3]
    p = predicted_field[:, 3:4]

    # torch.autograd.grad
    ones = torch.ones_like(p)

    grad_p = torch.autograd.grad(p, pos, grad_outputs=ones, create_graph=True)[0]
    dp_dx, dp_dy, dp_dz = grad_p[:, 0], grad_p[:, 1], grad_p[:, 2]

    grad_u = torch.autograd.grad(u, pos, grad_outputs=ones, create_graph=True)[0]
    du_dx, du_dy, du_dz = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]

    grad_v = torch.autograd.grad(v, pos, grad_outputs=ones, create_graph=True)[0]
    dv_dx, dv_dy, dv_dz = grad_v[:, 0], grad_v[:, 1], grad_v[:, 2]

    grad_w = torch.autograd.grad(w, pos, grad_outputs=ones, create_graph=True)[0]
    dw_dx, dw_dy, dw_dz = grad_w[:, 0], grad_w[:, 1], grad_w[:, 2]

    du_dxx = torch.autograd.grad(du_dx, pos, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 0]
    du_dyy = torch.autograd.grad(du_dy, pos, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0][:, 1]
    du_dzz = torch.autograd.grad(du_dz, pos, grad_outputs=torch.ones_like(du_dz), create_graph=True)[0][:, 2]
    laplacian_u = du_dxx + du_dyy + du_dzz

    dv_dxx = torch.autograd.grad(dv_dx, pos, grad_outputs=torch.ones_like(dv_dx), create_graph=True)[0][:, 0]
    dv_dyy = torch.autograd.grad(dv_dy, pos, grad_outputs=torch.ones_like(dv_dy), create_graph=True)[0][:, 1]
    dv_dzz = torch.autograd.grad(dv_dz, pos, grad_outputs=torch.ones_like(dv_dz), create_graph=True)[0][:, 2]
    laplacian_v = dv_dxx + dv_dyy + dv_dzz

    dw_dxx = torch.autograd.grad(dw_dx, pos, grad_outputs=torch.ones_like(dw_dx), create_graph=True)[0][:, 0]
    dw_dyy = torch.autograd.grad(dw_dy, pos, grad_outputs=torch.ones_like(dw_dy), create_graph=True)[0][:, 1]
    dw_dzz = torch.autograd.grad(dw_dz, pos, grad_outputs=torch.ones_like(dw_dz), create_graph=True)[0][:, 2]
    laplacian_w = dw_dxx + dw_dyy + dw_dzz

    # Continuity equation (conservation of mass)
    continuity_residual = du_dx + dv_dy + dw_dz

    # Momentum equation (ignoring gravity)
    momentum_x = rho * (u.squeeze() * du_dx + v.squeeze() * du_dy + w.squeeze() * du_dz) + dp_dx - mu * laplacian_u
    momentum_y = rho * (u.squeeze() * dv_dx + v.squeeze() * dv_dy + w.squeeze() * dv_dz) + dp_dy - mu * laplacian_v
    momentum_z = rho * (u.squeeze() * dw_dx + v.squeeze() * dw_dy + w.squeeze() * dw_dz) + dp_dz - mu * laplacian_w

    # Calculate the mean squared error of the residual as the loss
    loss_continuity = torch.mean(continuity_residual ** 2)
    loss_momentum = torch.mean(momentum_x ** 2) + torch.mean(momentum_y ** 2) + torch.mean(momentum_z ** 2)

    return loss_continuity + loss_momentum


def cvae_boundary_driven_loss(predicted_solution, true_solution, mu, log_var, query_pos_batch, query_pos):
    """
    Loss = w1 * OT_Loss + w2 * KLD_Loss + w3 * PDE_Loss
    """
    # --- 1. OT loss ---
    ot_loss = 0.0
    unique_batches = torch.unique(query_pos_batch)
    for batch_idx in unique_batches:
        mask = (query_pos_batch == batch_idx)
        if mask.sum() > 0:
            ot_loss += ot_loss_func(predicted_solution[mask], true_solution[mask])
    ot_loss /= len(unique_batches)

    # --- 2. KL loss ---
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kld_loss = torch.mean(kld_loss)

    # --- 3. PDE loss ---
    query_pos.requires_grad_(True)
    pde_loss = compute_ns_residuals(predicted_solution, query_pos)
    query_pos.requires_grad_(False)

    # --- 4. Total_loss ---
    total_loss = (config.LAMBDA_OT * ot_loss +
                  config.BETA_VAE * kld_loss +
                  config.LAMBDA_PDE * pde_loss)

    return total_loss, ot_loss, kld_loss, pde_loss