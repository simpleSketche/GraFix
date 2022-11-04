import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import batched_negative_sampling


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.conv_layers.append(GCNConv(input_dim, hidden_dim, aggr='add'))
            else:
                self.conv_layers.append(GCNConv(hidden_dim, hidden_dim, aggr='add'))
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

    


EPS = 1e-15
MAX_LOGSTD = 10


class VariationalGCNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 2 * output_dim)
        self.conv_mu = GCNConv(2 * output_dim, output_dim)
        self.conv_logstd = GCNConv(2 * output_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class InnerProductDecoder(nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        # print("In inner product decoder:")
        # print(value)
        # print()
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class EdgeDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, z, edge_index, sigmoid=True):
        head = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        logit = self.mlp(head).squeeze()
        return torch.sigmoid(logit) if sigmoid else logit



class GAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
    
    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, true_edge_type):
        # binary cross entropy loss for edge type prediction
        pred_edge_type = self.decoder(z, pos_edge_index, sigmoid=True)
        pos_case = (true_edge_type == 1)
        neg_case = (true_edge_type == 0)
        pos_loss = -torch.log(pred_edge_type[pos_case] + EPS).mean()
        neg_loss = -torch.log(1 - pred_edge_type[neg_case] + EPS).mean()
        
        # print("pred:", pred_edge_type)
        # print("true:", true_edge_type)
        # print()

        # print(f"pos: {pos_loss}, neg: {neg_loss}")

        return pos_loss + neg_loss
        
        # return bce_loss


        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

    @torch.no_grad()
    def test(self, z, pos_edge_index, true_edge_type):
        # calculate acc
        pred_edge_type = self.decoder(z, pos_edge_index, sigmoid=True)
        pred_edge_type = torch.round(pred_edge_type)
        # print("pred_edge_type:")
        # print(pred_edge_type)
        acc = torch.sum(pred_edge_type == true_edge_type) / true_edge_type.size(0)
        return acc



# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/autoencoder.html#VGAE
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py
class VGAE(GAE):
    def __init__(self, encoder, decoder=None):
        super().__init__(encoder, decoder)
    
    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) + torch.exp(logstd)
        else:
            return mu
    
    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu=None, logstd=None):
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

