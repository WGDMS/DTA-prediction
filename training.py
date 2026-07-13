import sys
import os
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

from model import GraphModel
from utils import *
from dataprocessing import create_dataset
from config import get_config, checkpoint_path


dataset_arg = sys.argv[1]
split_arg = sys.argv[2]

cfg = get_config(dataset_arg, split_arg)

dataset = cfg["dataset"]
split = cfg["split"]
LR = cfg["lr"]
TRAIN_BATCH_SIZE = cfg["batch_size"]
TEST_BATCH_SIZE = cfg["test_batch_size"]
NUM_EPOCHS = cfg["epochs"]
PATIENCE = cfg["patience"]
seeds = cfg["seeds"]

print(f"Dataset: {dataset}")
print(f"Split strategy: {split}")
print(f"Number of epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LR}")
print(f"Batch size: {TRAIN_BATCH_SIZE}")
print(f"Early stopping patience: {PATIENCE}")

models_dir = 'models'
results_dir = 'results'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
print(f"Training with dataset {dataset}")

torch.cuda.empty_cache()
train_data, valid_data = create_dataset(dataset, split)

for seed in seeds:
    print(f"Running training for seed {seed}...")
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize model
    model = GraphModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

   # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=5e-4)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)
    
    best_mse = float('inf')
    patience_counter = 0
    torch.cuda.empty_cache()
    train_losses=[]
    val_losses=[]
    
    for epoch in range(NUM_EPOCHS):
        tr_loss = train(model, device, train_loader, optimizer, epoch + 1)
        train_losses.append(tr_loss)
        with torch.no_grad():
            G, P = predicting(model, device, valid_loader)
        val_loss = get_mse(G, P)
        val_losses.append(val_loss)
        
        if val_loss < best_mse:
            best_mse = val_loss
            patience_counter = 0
            model_filename = checkpoint_path(dataset, split, seed)
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved for seed {seed}, epoch {epoch+1}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1} for seed {seed}")
            break

       # scheduler.step()
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label=f'Train Loss')
    plt.plot(val_losses, label=f'Val Loss', linestyle='--')
    plt.title(f'Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plot_filename = f'{results_dir}/train_val_loss_plot_seed{seed}.png'
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory

print("Training completed for all seeds-DTA")
