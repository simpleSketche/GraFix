import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.conv_layers.append(tgnn.GCNConv(input_dim, hidden_dim, aggr='add'))
            else:
                self.conv_layers.append(tgnn.GCNConv(hidden_dim, hidden_dim, aggr='add'))
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear(x)

        # node level prediction (rotation)
        node_y = self.sigmoid(x)
        # edge level prediction (connection type)

        return node_y