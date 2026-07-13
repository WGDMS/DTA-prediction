import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
#%matplotlib inline
#import matplotlib.pyplot as plt
import numpy as np
import subprocess
from math import sqrt
from sklearn.metrics import average_precision_score
from scipy import stats
import time

class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',y=None, transform=None,
                 pre_transform=None, mol_graph=None, drug_key=None, target_key=None, target_graph=None):

        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.process(drug_key, target_key, y, mol_graph, target_graph)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, drug_key, target_key, y, mol_graph, target_graph):
        #the three lists must be the same length
        assert (len(drug_key) == len(target_key) and len(drug_key) == len(y))
        data_list_mol = []
        data_list_pro = []
        data_list_fun = []
        data_len = len(drug_key)

        
        for i in range(data_len):
            #smiles = xd[i]
            tar_key = target_key[i]
            d_key = str(drug_key[i])
            labels = y[i]
                            
            atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list= mol_graph[d_key]    
      
            target_size, target_features, target_edge_index = target_graph[tar_key]
        
            GINE_mol = DATA.Data(x=torch.Tensor(atom_features),
                                    edge_index=torch.LongTensor(bond_list).reshape(-1, 2).transpose(1, 0),
                                    edge_attr=torch.Tensor(bond_features).reshape(-1, 11),
                                    y=torch.FloatTensor([labels]))
            #GINE_mol.__setitem__('c_size', torch.LongTensor([c_size]))
            fg_features_tensor = torch.Tensor(fg_features)
            GINE_fun = DATA.Data(fg_x=torch.Tensor(fg_features),
                                    fg_edge_index=torch.LongTensor(fg_edge_list).reshape(-1, 2).transpose(1, 0),
                                    fg_edge_attr=torch.Tensor(fg_edge_features).reshape(-1, 101),
                                    y=torch.FloatTensor([labels]),
                                    num_nodes=fg_features_tensor.size(0))
            
            
            #target_masks= torch.Tensor(target_masks), 
            GIN_pro = DATA.Data( x=torch.Tensor(target_features),
                                    edge_index=torch.LongTensor(target_edge_index).transpose(1, 0), 
                                  
                                    y=torch.FloatTensor([labels]))
            GIN_pro.__setitem__('target_size', torch.LongTensor([target_size]))
          
            
            
            
            data_list_mol.append(GINE_mol)
            data_list_pro.append(GIN_pro)
            data_list_fun.append(GINE_fun)     
       
        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]
            data_list_fun = [data for data in data_list_fun if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]
            data_list_fun = [self.pre_transform(data) for data in data_list_fun]
        self.data_mol = data_list_mol
        self.data_pro = data_list_pro
        self.data_fun = data_list_fun
        
    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx], self.data_fun[idx]

def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss and Validation Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/sample_data/loss_plot.png')
    plt.show()
        
# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 10
    
    current_batch_size = data_mol.num_graphs
    loss_fn = torch.nn.MSELoss()
    train_loss=0.0
    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        data_fun = data[2].to(device)
        optimizer.zero_grad()
        current_batch_size = data_mol.num_graphs
        output = model(data_mol, data_pro, data_fun)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * current_batch_size,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                          loss.item()))
    train_loss/=len(train_loader)
    print(f'\nEpoch {epoch}: Average Training Loss: {train_loss:.6f}')

    return train_loss

# predict
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            data_fun = data[2].to(device)
            output = model(data_mol, data_pro, data_fun)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

# defined the required metrics 

def get_aupr(Y, P, threshold=7.0):
    # print(Y.shape,P.shape)
    Y = np.where(Y >= 7.0, 1, 0)
    P = np.where(P >= 7.0, 1, 0)
    aupr = average_precision_score(Y, P)
    return aupr


def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair != 0:
        return summ / pair
    else:
        return 0


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def get_rm2(ys_orig, ys_pred):
    r2 = r_squared_error(ys_orig, ys_pred)
    r02 = squared_error_zero(ys_orig, ys_pred)
    return r2 * (1.0 - np.sqrt(np.abs(r2 - r02)))


def get_rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def get_pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def get_spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def get_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


#prepare the protein and drug pairs
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    return batchA, batchB, batchC
