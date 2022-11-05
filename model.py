import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_geometric.nn import GCNConv    
from utils import R2_score

EPS = 1e-15
MAX_LOGSTD = 10


class VariationalGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # maybe change mu, logstd to linear layer
        # self.conv_mu = GCNConv(hidden_dim, hidden_dim)
        # self.conv_logstd = GCNConv(hidden_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, hidden_dim)
        self.linear_logstd = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        # return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        return self.linear_mu(x), self.linear_logstd(x)


class VariationalGraphDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, edge_index):
        z = self.linear_1(z).relu()
        z = self.conv1(z, edge_index).relu()
        z = self.linear_2(z)
        return z



class VGAE(nn.Module):
    def __init__(self, encoder, decoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
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
    
    def recon_loss(self, z, edge_index, x):
        # print("start reconstructing loss")
        # target: reconstruct location (x, y) for boxes
        true_location = x[:, 3:5]
        pred_location = self.decoder(z, edge_index)
        loss = F.mse_loss(pred_location, true_location)
        # print("loss:", loss)
        # print()
        return loss

    def kl_loss(self, mu=None, logstd=None):
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    @torch.no_grad()
    def test(self, z, edge_index, x):
        # calculate R2 score
        true_location = x[:, 3:5]
        pred_location = self.decoder(z, edge_index)
        acc = R2_score(pred_location, true_location)
        return acc