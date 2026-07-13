import os
import sys
import torch
import numpy as np
from utils import *
from scipy import stats
from model import GraphModel

from dataprocessing import create_dataset_for_test
import matplotlib.pyplot as plt
import seaborn as sns
from config import get_config, checkpoint_path, result_path

def plot_true_vs_predicted(true_values, predicted_values, dataset, seed=None):
    """
    Create a scatter plot with regression line for true vs predicted values
    """
    plt.figure(figsize=(10, 8))
    
    # Scatter + regression line
    sns.regplot(x=true_values, y=predicted_values, 
                scatter_kws={'alpha':0.6, 's':30}, 
                line_kws={'color':'red', 'linewidth':2})
    
    # Regression statistics
    slope, intercept, r_value, p_value, std_err = stats.linregress(true_values, predicted_values)
    mse = np.mean((true_values - predicted_values) ** 2)
    rmse = np.sqrt(mse)
    
    # Add ideal y=x line
    min_val = min(np.min(true_values), np.min(predicted_values))
    max_val = max(np.max(true_values), np.max(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Ideal fit (y=x)')
    
    plt.xlabel('True Affinity Values', fontsize=12)
    plt.ylabel('Predicted Affinity Values', fontsize=12)
    plt.title(f'True vs Predicted Affinity - {dataset} Dataset', fontsize=14)
    
    # Add small text box with metrics
    textstr = f'MSE = {mse:.4f}\nSlope = {slope:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top', bbox=props)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def plot_kde_affinity(true_values, predicted_values, dataset, seed=None):
    """
    Create Kernel Density Estimate plot: Affinity score vs. Probability density
    """
    plt.figure(figsize=(10, 8))
    
    # KDE plots for true and predicted
    sns.kdeplot(true_values, color='blue', label='True Affinity', linewidth=2, fill=True, alpha=0.3)
    sns.kdeplot(predicted_values, color='red', label='Predicted Affinity', linewidth=2, fill=True, alpha=0.3)
    
    plt.xlabel('Affinity Score', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    #plt.title(f'Affinity Score Distribution - {dataset} Dataset', fontsize=14)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            data_clique = data[2].to(device)
            output = model(data_mol, data_pro, data_clique)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()



dataset_arg = sys.argv[1]
split_arg = sys.argv[2]

cfg = get_config(dataset_arg, split_arg)

dataset = cfg["dataset"]
split = cfg["split"]
TEST_BATCH_SIZE = cfg["test_batch_size"]
seeds = cfg["seeds"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

models_dir = 'models'
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

test_data = create_dataset_for_test(dataset, split)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

all_metrics = []

for seed in seeds:
    print(f"Evaluating model for seed {seed}...")
    
    model = GraphModel().to(device)
    model_file_name = checkpoint_path(dataset, split, seed)
    model.load_state_dict(torch.load(model_file_name))
    model.to(device)
    model.eval()

    # Predict
    G, P = predicting(model, device, test_loader)

    # Compute metrics
    mse = get_mse(G, P)
    rmse = get_rmse(G, P)
    pearson = get_pearson(G, P)
    spearman = get_spearman(G, P)
    cindex = get_cindex(G, P)
    rm2 = get_rm2(G, P)
    print(model_file_name)
    all_metrics.append([mse, rmse, pearson, spearman, cindex, rm2])


# Compute overall stats
all_metrics = np.array(all_metrics)
mean_metrics = np.mean(all_metrics, axis=0)
std_metrics = np.std(all_metrics, axis=0)

print("\nFinal Results Across 5 Runs:")
metric_names = ["MSE", "RMSE", "Pearson", "Spearman", "C-Index", "RM2"]
for i, name in enumerate(metric_names):
    print(f"{name}: Mean = {mean_metrics[i]:.4f}, Std = {std_metrics[i]:.4f}")

# Save results
os.makedirs('results', exist_ok=True)
result_file = result_path(dataset, split)
with open(result_file, 'w') as f:
    f.write("Final Results Across 5 Runs:\n")
    for i, name in enumerate(metric_names):
        f.write(f"{name}: Mean = {mean_metrics[i]:.4f}, Std = {std_metrics[i]:.4f}\n")

print(f"Results saved to {result_file}")
