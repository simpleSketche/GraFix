"""
I build VAE by taking these code as example
https://github.com/chendaichao/VAE-pytorch/blob/master/Models/VAE/model.py
https://github.com/timbmg/VAE-CVAE-MNIST/blob/master/models.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_geometric.nn import GCNConv    



class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.dnn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(tgnn.GCNConv(hidden_dim, hidden_dim, aggr='add'))

        self.dnn2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )


    def forward(self, x, edge_index):
        x = self.dnn1(x)
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
        x = self.dnn2(x)
        return x




