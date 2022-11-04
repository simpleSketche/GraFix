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
        print("In inner product decoder:")
        print(value)
        print()
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class GAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
    
    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, batch, neg_edge_index=None):
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        if neg_edge_index is None:
            neg_edge_index = batched_negative_sampling(pos_edge_index, batch, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        # print("pos logits:", self.decoder(z, pos_edge_index, sigmoid=True))
        # print("neg loss:", self.decoder(z, neg_edge_index, sigmoid=True))
        # print()
        return pos_loss + neg_loss * 100

    @torch.no_grad()
    def test(self, z, pos_edge_index, batch, neg_edge_index=None):
        # calculate pos_acc, neg_acc
        if neg_edge_index is None:
            neg_edge_index = batched_negative_sampling(pos_edge_index, batch, z.size(0))

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        pos_pred = torch.round(self.decoder(z, pos_edge_index, sigmoid=True))
        neg_pred = torch.round(self.decoder(z, neg_edge_index, sigmoid=True))
        pos_acc = torch.sum((pos_pred == pos_y)) / pos_y.size(0)
        neg_acc = torch.sum((neg_pred == neg_y)) / neg_y.size(0)

        return pos_acc, neg_acc


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/autoencoder.html#VGAE
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py
class VGAE(GAE):
    def __init__(self, encoder, decoder=None):
        super().__init__(encoder, decoder)
    
    def reparametrize(self, mu, logstd):
        if self.training:
            # return mu + torch.randn_like(logstd) + torch.exp(logstd)
            z = mu + torch.randn_like(logstd) + torch.exp(logstd)
            # print("mu:", mu)
            # print("logstd:", logstd)
            # print("z:", z)
            # print()
            return z
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

