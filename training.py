import sys
import os
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader

from model import GraphModel
from utils import *
from dataprocessing import create_dataset

# Dataset
dataset = ['davis', 'kiba'][int(sys.argv[1])]
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.001
NUM_EPOCHS = int(sys.argv[2])
split = sys.argv[3]  # This will take 'S1', 'S2', 'S3', or 'S4'
PATIENCE = 500  # Early stopping patience


print(f"Dataset: {dataset}")
print(f"Number of epochs: {num_epochs}")
print(f"Split strategy: {split}")

#gnn_type = ["GIN", "GAT", "GCN", "GAT_GCN"][int(sys.argv[3])]
#print(f"Using GNN type: {gnn_type}")

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

for seed in range(5):
    print(f"Running training for seed {seed}...")
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize model
    model = GraphModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=5e-4)
    
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
            model_filename = f'{models_dir}/model_{dataset}_seed{seed}.model'
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved for seed {seed}, epoch {epoch+1}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1} for seed {seed}")
            break

        scheduler.step()
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
