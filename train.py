# train.py
import torch
import os
import numpy as np
from tqdm import tqdm

import config
from model import PGAF
from mydata import get_dataloaders_with_split
from loss_function import cvae_boundary_driven_loss


# --- EarlyStopping ---
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
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
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {self.val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# --- Training and Valid ---
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    loop = tqdm(dataloader, leave=False)
    for data in loop:
        data = data.to(device)
        reconstructed_solution, mu, log_var = model(data)
        loss, ot_loss, kld_loss, pde_loss = cvae_boundary_driven_loss(
            reconstructed_solution, data.y, mu, log_var, data.query_pos_batch, data.query_pos
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item(), ot=ot_loss.item(), kld=kld_loss.item(), pde=pde_loss.item())


def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            model.train()
            reconstructed_solution, mu, log_var = model(data)
            model.eval()
            loss, _, _, _ = cvae_boundary_driven_loss(
                reconstructed_solution, data.y, mu, log_var, data.query_pos_batch, data.query_pos
            )
            total_val_loss += loss.item()

    return total_val_loss / len(dataloader)


if __name__ == "__main__":
    print(f"Using device: {config.DEVICE}")

    try:
        train_loader, val_loader, _ = get_dataloaders_with_split(
            root_dir=config.DATA_DIR,
            batch_size=config.BATCH_SIZE,
            val_split_ratio=config.VALIDATION_SPLIT_RATIO,  # 定义验证集比例
            num_workers=config.NUM_WORKERS
        )
    except FileNotFoundError as e:
        print(e)
        exit()

    model = PGAF().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=True,
        path=config.MODEL_SAVE_PATH
    )

    print("Starting training with dynamic validation split...")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"--- Epoch {epoch}/{config.NUM_EPOCHS} ---")
        train_one_epoch(model, train_loader, optimizer, config.DEVICE)

        avg_val_loss = validate_one_epoch(model, val_loader, config.DEVICE)
        print(f"Average Validation Loss: {avg_val_loss:.6f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print(f"\nTraining complete. Best model saved to {config.MODEL_SAVE_PATH}")