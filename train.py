# train.py
import torch
import os
import numpy as np
from tqdm import tqdm

# Import project-specific modules
import config
from model import PGAF
from mydata import get_dataloaders_with_split
from loss_function import cvae_boundary_driven_loss


# --- EarlyStopping Class ---
class EarlyStopping:
    """A utility class to implement early stopping and save the best model."""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): A function to use for printing messages (e.g., print or a logger).
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """
        Call method to check if training should be stopped.
        Saves the model if validation loss decreases.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        # Ensure the directory for the saved model exists.
        save_dir = os.path.dirname(self.path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# --- Training and Validation Functions ---
def train_one_epoch(model, dataloader, optimizer, device):
    """Performs a single training epoch."""
    model.train()  # Set the model to training mode
    loop = tqdm(dataloader, desc="Training", leave=True)
    total_loss, total_ot, total_kld, total_pde = 0, 0, 0, 0

    for data in loop:
        data = data.to(device)
        
        # [CRITICAL FIX] Enable gradient computation for query_pos.
        # This is necessary for calculating the PDE loss, which requires derivatives with respect to position.
        data.query_pos.requires_grad_(True)
        
        # Forward pass: The model's forward method handles the CVAE logic.
        reconstructed_solution, mu, log_var = model(data)
        
        # Calculate loss
        loss, ot_loss, kld_loss, pde_loss = cvae_boundary_driven_loss(
            reconstructed_solution, data.y, mu, log_var, data.query_pos_batch, data.query_pos
        )
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record losses and update the progress bar description.
        total_loss += loss.item()
        total_ot += ot_loss.item()
        total_kld += kld_loss.item()
        total_pde += pde_loss.item()
        loop.set_postfix(loss=loss.item(), ot=ot_loss.item(), kld=kld_loss.item(), pde=pde_loss.item())
        
    avg_loss = total_loss / len(dataloader)
    avg_ot = total_ot / len(dataloader)
    avg_kld = total_kld / len(dataloader)
    avg_pde = total_pde / len(dataloader)
    print(f"Avg Train Loss: {avg_loss:.6f} (OT: {avg_ot:.6f}, KLD: {avg_kld:.6f}, PDE: {avg_pde:.6f})")


def validate_one_epoch(model, dataloader, device):
    """Performs a single validation epoch."""
    model.eval()  # Set the model to evaluation mode
    loop = tqdm(dataloader, desc="Validating", leave=True)
    total_val_loss = 0
    
    # [CRITICAL FIX] Validation logic for a complex CVAE/PINN model.
    # We don't update weights, but to calculate the full loss (including KLD and PDE),
    # we need to temporarily enable gradients and force the model to use its training path.
    for data in loop:
        data = data.to(device)
        
        # Temporarily enable the gradient context for loss calculation only.
        with torch.enable_grad():
            data.query_pos.requires_grad_(True)
            
            # Temporarily switch the model to train() mode to ensure the CVAE uses the posterior encoder path,
            # which is necessary to compute mu and log_var for the KLD loss.
            model.train()
            reconstructed_solution, mu, log_var = model(data)
            model.eval() # Immediately switch back to evaluation mode after the forward pass.
            
            # Calculate the full validation loss, including the PDE component.
            loss, _, _, _ = cvae_boundary_driven_loss(
                reconstructed_solution, data.y, mu, log_var, data.query_pos_batch, data.query_pos,
                compute_pde=True # Explicitly compute PDE loss for validation.
            )
        
        total_val_loss += loss.item()
        loop.set_postfix(val_loss=loss.item())
        
    avg_val_loss = total_val_loss / len(dataloader)
    return avg_val_loss


# --- Main Execution Block ---
if __name__ == "__main__":
    # For safety with multiprocessing data loading on Windows or macOS.
    # It's usually not needed on Linux.
    # torch.multiprocessing.set_start_method('spawn') 
    
    print(f"Using device: {config.DEVICE}")

    # Load data
    try:
        train_loader, val_loader, test_loader = get_dataloaders_with_split(
            root_dir=config.DATA_DIR,
            batch_size=config.BATCH_SIZE,
            val_split_ratio=config.VALIDATION_SPLIT_RATIO,
            num_workers=config.NUM_WORKERS
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your data directory is correctly set up in config.py and contains the 'train' folder.")
        exit()

    # Initialize model, optimizer, and early stopping handler
    model = PGAF().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=True,
        path=config.MODEL_SAVE_PATH
    )

    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    # Main training loop
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.NUM_EPOCHS} ---")
        
        # Train for one epoch
        train_one_epoch(model, train_loader, optimizer, config.DEVICE)
        
        # Validate the model
        avg_val_loss = validate_one_epoch(model, val_loader, config.DEVICE)
        print(f"Average Validation Loss: {avg_val_loss:.6f}")
        
        # Check for early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered due to no improvement in validation loss.")
            break

    print("\n" + "="*50)
    print("Training complete.")
    print(f"Best model saved to {config.MODEL_SAVE_PATH}")
    print("="*50)
