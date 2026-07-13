import os, sys, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv, GINConv, GINEConv,  global_max_pool, global_add_pool, global_mean_pool,  GlobalAttention
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.utils import dropout_adj, softmax
#from torch_geometric.nn import AttentiveFP
#from torch_scatter import scatter
import math
import numpy as np
dtype = torch.float32

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GraphModel(torch.nn.Module):
    def __init__(self, n_output=1, output_dim=128, dropout=0.2):
        super(GraphModel, self).__init__()

        print('GraphModel Loaded')
        self.n_output = n_output

         # GIN layers for protein

        self.MolAtomGINE = MolAtomGINE(dropout=0.1)
        self.MolSubGINE = MolSubGINE(dropout=0.1)
        #self.ProGIN = ProGIN(dropout=0.2)
       
        self.ProGIN = ProGIN(num_features_pro=54, hidden_dim=128, output_dim=128, num_layers=3, dropout=0.2)
        
        self.featureFusion = MolAttFusion(hidden_dim=output_dim, num_heads=8, dropout=dropout)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * 2* output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro, data_mot):
        # get graph input
        
        #mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        #mot_x, mot_edge_index, mot_batch = data_mot.x, data_mot.edge_index, data_mot.batch
        # target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        
        xa = self.MolAtomGINE(data_mol)   # [B, 128]
        xf = self.MolSubGINE(data_mot)    # [B, 128]
        xt = self.ProGIN(data_pro)        # [B, 256]

        x = self.featureFusion(xa, xf)    # [B, 256]
        # print("x.shape", x.shape)    
        # print("xt.shape", xt.shape)   
        xcon = torch.cat((x, xt), dim=1)  # [B, 512]

      
        # dense layers
        xcon = self.fc1(xcon)
        xcon = self.relu(xcon)
        xcon = self.dropout(xcon)
        xcon = self.fc2(xcon)
        xcon = self.relu(xcon)
        xcon = self.dropout(xcon)
        out = self.out(xcon)
        return out

class MolAtomGINE(nn.Module):
    def __init__(self, num_atom_layers=3, latent_dim=128,
                 atom_dim=101,bond_dim=11,
                pool='mean', dropout=0, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_atom_layers = num_atom_layers


        # embedding
        self.atom_embedding = nn.Linear(atom_dim, latent_dim)
        
        self.bond_embedding = nn.ModuleList(
            [nn.Linear(bond_dim, latent_dim) for _ in range(num_atom_layers)]
        )


        # gnn
        self.atom_gin = nn.ModuleList(
            [GINEConv(
            nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
            ),
            eps=0.0,
            train_eps=False
            ) for _ in range(num_atom_layers)]
            )
        self.atom_bn = nn.ModuleList(
            [nn.BatchNorm1d(latent_dim) for _ in range(num_atom_layers)]
        )

        
        # pooling
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'sum':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling!")

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        

    def forward(self, data):
        atom_x, atom_edge_index, atom_edge_attr, atom_batch = data.x, data.edge_index, data.edge_attr, data.batch
       

        # one-hot to vec
        atom_x = self.atom_embedding(atom_x)
       

        # atom-level gnn
        for i in range(self.num_atom_layers):
            atom_x = self.atom_gin[i](atom_x, atom_edge_index, self.bond_embedding[i](atom_edge_attr))
            atom_x = self.atom_bn[i](atom_x)
            if i != self.num_atom_layers-1:
                atom_x = self.relu(atom_x)
            atom_x = self.dropout(atom_x)


        atom_graph = self.pool(atom_x, atom_batch)
       

        return atom_graph


        
class MolSubGINE(nn.Module):
    def __init__(self,num_fg_layers=2, latent_dim=128,
                 fg_dim=73,fg_edge_dim=101,
                 pool='mean', dropout=0, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_fg_layers = num_fg_layers
        

        # embedding
        
        self.fg_embedding = nn.Linear(fg_dim, latent_dim)

        self.fg_edge_embedding = nn.ModuleList(
            [nn.Linear(fg_edge_dim, latent_dim) for _ in range(num_fg_layers)]
        )

        # gnn
        self.fg_gin = nn.ModuleList(
            [GINEConv(
            nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
            ),
            eps=0.0,
            train_eps=False
            ) for _ in range(num_fg_layers)]
        )
        self.fg_bn = nn.ModuleList(
            [nn.BatchNorm1d(latent_dim) for _ in range(num_fg_layers)]
        )
        
        # pooling
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'sum':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling")

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    def forward(self, data):
        fg_x, fg_edge_index, fg_edge_attr, fg_batch = data.fg_x, data.fg_edge_index, data.fg_edge_attr, data.batch
        
        # one-hot to vec
        
        fg_x = self.fg_embedding(fg_x)

        
        #671,128      

        # fg-level gnn
        for i in range(self.num_fg_layers):
            fg_x = self.fg_gin[i](fg_x, fg_edge_index, self.fg_edge_embedding[i](fg_edge_attr))
            fg_x = self.fg_bn[i](fg_x)
            if i != self.num_fg_layers-1:
                fg_x = self.relu(fg_x)
            fg_x = self.dropout(fg_x)


        fg_graph = self.pool(fg_x, fg_batch)

       
       
        return fg_graph


class ProGIN(nn.Module):
    def __init__(
        self,
        num_features_pro=54,
        hidden_dim=128,
        output_dim=128,
        num_layers=3,
        dropout=0.2):
        super(ProGIN, self).__init__()

        self.num_layers = num_layers

        # 54-dimensional residue features into the 128-dimensional hidden space
        self.input_projection = nn.Linear(num_features_pro, hidden_dim)

        # Protein GIN layers: all layers preserve the 128-dimensional hidden size
        self.pro_gin = nn.ModuleList()
        self.pro_bn = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )

            self.pro_gin.append(
                GINConv(
                    mlp,
                    eps=0.0,
                    train_eps=False
                )
            )

            self.pro_bn.append(nn.BatchNorm1d(hidden_dim))

        # Fully connected refinement layers
        # After global mean pooling, the protein representation is [B, 128].
        # These layers produce the final protein vector [B, 256].
        self.pro_fc_g1 = nn.Linear(hidden_dim, 1024)
        self.pro_fc_g2 = nn.Linear(1024, 2 * output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        target_x, target_edge_index, target_batch = (
            data.x,
            data.edge_index,
            data.batch
        )

        # [num_residues, 54] -> [num_residues, 128]
        xt = self.input_projection(target_x)

        for i in range(self.num_layers):
            xt = self.pro_gin[i](xt, target_edge_index)
            xt = self.pro_bn[i](xt)

            if i != self.num_layers - 1:
                xt = self.relu(xt)

            xt = self.dropout(xt)

        # [B, 128]
        xt = global_mean_pool(xt, target_batch)

        # [B, 128] -> [B, 1024] -> [B, 256]
        xt = self.pro_fc_g1(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)

        xt = self.pro_fc_g2(xt)

        return xt


class MolAttFusion(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=8, dropout=0.2):
        super(MolAttFusion, self).__init__()

        self.fg_projection = nn.Linear(hidden_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.output_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, h_atom, h_fg, return_attention=False):
        h_fg_projected = self.fg_projection(h_fg)

        # [B, 2, d]
        h_mol = torch.stack(
            [h_atom, h_fg_projected],
            dim=1
        )

        h_attn, attn_weights = self.attention(
            h_mol,
            h_mol,
            h_mol,
            need_weights=True,
            average_attn_weights=False
        )

        h_refined = self.layer_norm(
            h_mol + self.dropout(h_attn)
        )

        # [B, 2d] = [B, 256]
        h_out = self.output_mlp(
            h_refined.reshape(h_refined.size(0), -1)
        )

        if return_attention:
            return h_out, attn_weights

        return h_out

        
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        values = values.reshape(N, -1, self.heads, self.head_dim)
        keys = keys.reshape(N, -1, self.heads, self.head_dim)
        queries = query.reshape(N, -1, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, -1, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out
