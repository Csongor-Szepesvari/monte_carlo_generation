"""
Top-K Predictor - Deep Learning Model for Top-K Sum Prediction

This module implements a deep learning model based on Deep Sets architecture to predict
the sum of top-k elements from a distribution. The model is designed to handle variable
input sizes and different k values.

The architecture consists of:
1. A permutation-invariant function (phi) that processes each element independently
2. A summation operation to aggregate the processed elements
3. A final network (rho) that takes the aggregated representation and k value to predict the sum

The model is trained using PyTorch and includes utilities for data loading, training,
evaluation, and visualization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class PhiMLP(nn.Module):
    """
    Permutation-invariant network that processes each element independently.
    
    This network maps each input element (mu, sigma, n) to a higher-dimensional
    representation that captures the element's contribution to the top-k sum.
    
    Args:
        input_dim (int): Dimension of each input element (default: 3 for mu, sigma, n)
        hidden_dim (int): Size of the hidden layer (default: 32)
        output_dim (int): Size of the output representation (default: 32)
    """
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=32):
        super(PhiMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU()
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_elements, input_dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_elements, output_dim]
        """
        return self.net(x)

class RhoMLP(nn.Module):
    """
    Final network that processes the aggregated representation and k value.
    
    This network takes the sum of element representations and the normalized k value
    to predict the sum of the top-k elements.
    
    Args:
        input_dim (int): Dimension of the input (sum of element representations + k value)
                         (default: 33 = 32 + 1)
    """
    def __init__(self, input_dim=33):
        super(RhoMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, z, k):
        """
        Forward pass through the network.
        
        Args:
            z (torch.Tensor): Aggregated representation tensor of shape [batch_size, output_dim]
            k (torch.Tensor): Normalized k value tensor of shape [batch_size, 1]
            
        Returns:
            torch.Tensor: Predicted top-k sum of shape [batch_size, 1]
        """
        zk = torch.cat([z, k], dim=-1)
        return self.net(zk)

class DeepSetsTopKModel(nn.Module):
    """
    Deep Sets model for predicting the sum of top-k elements.
    
    This model implements the Deep Sets architecture, which consists of:
    1. A permutation-invariant function (phi) applied to each element
    2. A summation operation to aggregate the processed elements
    3. A final network (rho) that takes the aggregated representation and k value
    """
    def __init__(self):
        super(DeepSetsTopKModel, self).__init__()
        self.phi = PhiMLP(input_dim=3, hidden_dim=32, output_dim=32)
        self.rho = RhoMLP(input_dim=33)

    def forward(self, x, k):
        """
        Forward pass through the Deep Sets model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_elements, input_dim]
            k (torch.Tensor): Normalized k value tensor of shape [batch_size, 1]
            
        Returns:
            torch.Tensor: Predicted top-k sum of shape [batch_size]
        """
        phi_out = self.phi(x)
        summed = phi_out.sum(dim=1)
        out = self.rho(summed, k)
        return out.squeeze(-1)

class TopKDataset(Dataset):
    """
    Dataset for training and evaluating the top-k prediction model.
    
    This dataset processes a dataframe containing mu, sigma, n values for different
    distributions, along with the k value and the average sum of top-k elements.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the dataset
        k_max (int): Maximum k value for normalization (default: 60)
    """
    def __init__(self, dataframe):
        df = dataframe
        self.x = df[[f"mu{i+1}" for i in range(4)] + [f"sigma{i+1}" for i in range(4)] + [f"n{i+1}" for i in range(4)]].values.astype(np.float32)
        # Normalize k by dividing by total_n for each row
        self.k = df[["k"]].values.astype(np.float32) / df[["total_n"]].values.astype(np.float32)
        self.y = df[["avg_topk_sum"]].values.astype(np.float32)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (x_row, k_val, y_val) where:
                - x_row is a tensor of shape [4, 3] containing mu, sigma, n values
                - k_val is a tensor of shape [1] containing the normalized k value
                - y_val is a tensor of shape [1] containing the average top-k sum
        """
        x_row = self.x[idx].reshape(4, 3)
        k_val = self.k[idx]
        y_val = self.y[idx]
        return torch.tensor(x_row), torch.tensor(k_val), torch.tensor(y_val)

def train(model, dataloader, optimizer, loss_fn, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The model to train
        dataloader (DataLoader): DataLoader for the training set
        optimizer (torch.optim.Optimizer): Optimizer for training
        loss_fn (callable): Loss function
        device (torch.device): Device to use for training
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    for x, k, y in dataloader:
        x, k, y = x.to(device), k.to(device), y.to(device).squeeze(-1)
        optimizer.zero_grad()
        y_pred = model(x, k)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, loss_fn, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model (nn.Module): The model to evaluate
        dataloader (DataLoader): DataLoader for the evaluation set
        loss_fn (callable): Loss function
        device (torch.device): Device to use for evaluation
        
    Returns:
        tuple: (avg_loss, mse, mae, r2) where:
            - avg_loss is the average loss
            - mse is the mean squared error
            - mae is the mean absolute error
            - r2 is the R^2 score
    """
    model.eval()
    total_loss = 0
    y_true = []
    y_pred_all = []
    with torch.no_grad():
        for x, k, y in dataloader:
            x, k, y = x.to(device), k.to(device), y.to(device).squeeze(-1)
            preds = model(x, k)
            loss = loss_fn(preds, y)
            total_loss += loss.item() * x.size(0)
            y_true.append(y.cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred_all = np.concatenate(y_pred_all)
    mse = mean_squared_error(y_true, y_pred_all)
    mae = mean_absolute_error(y_true, y_pred_all)
    r2 = r2_score(y_true, y_pred_all)
    return total_loss / len(dataloader.dataset), mse, mae, r2

def plot_losses(train_losses, val_losses, output_path):
    """
    Plot training and validation losses.
    
    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        output_path (str): Path to save the plot
    """
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    # Configuration parameters
    batch_size = 64
    lr = 1e-3
    epochs = 100
    patience = 10
    csv_path = 'expanded_monte_carlo_topk.csv'
    log_dir = "runs/topk_experiment"
    os.makedirs(log_dir, exist_ok=True)

    # Set up device, model, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSetsTopKModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()
    writer = SummaryWriter(log_dir)

    # Load and split the dataset
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    # Create datasets and dataloaders
    train_dataset = TopKDataset(train_df)
    val_dataset = TopKDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop setup
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss, _, _, _ = evaluate(model, val_loader, loss_fn, device)
        epoch_time = time.time() - epoch_start

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Time/Epoch", epoch_time, epoch)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Time = {epoch_time:.2f}s")

        # Save best model and check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pt"))
            torch.jit.save(torch.jit.script(model), os.path.join(log_dir, "best_model.pt.jit"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    writer.close()

    # Report training results
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")

    # Plot and save loss curves
    plot_losses(train_losses, val_losses, os.path.join(log_dir, "loss_curve.png"))

    # Final evaluation on validation set
    model.load_state_dict(torch.load(os.path.join(log_dir, "best_model.pt")))
    _, mse, mae, r2 = evaluate(model, val_loader, loss_fn, device)
    print(f"Final evaluation on validation set:\n  MSE = {mse:.4f}\n  MAE = {mae:.4f}\n  R^2 = {r2:.4f}")
