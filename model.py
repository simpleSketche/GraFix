import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_geometric.nn import GCNConv    


class VariationalGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, hidden_dim)
        self.linear_logvar = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.linear_mu(x), self.linear_logvar(x)


class VariationalGraphDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, edge_index):
        z = self.linear_1(z).relu()
        z = self.conv1(z, edge_index).relu()
        z = self.linear_2(z)
        # first output is classification
        z[:, 0] = torch.sigmoid(z[:, 0])
        return z



class VGAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.hidden_dim == decoder.hidden_dim
        self.hidden_dim = encoder.hidden_dim
    
    def reparametrize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x, edge_index):
        mu, log_var = self.encoder(x, edge_index)
        z = self.reparametrize(mu, log_var)
        recon_x = self.decoder(z, edge_index)
        return recon_x, mu, log_var, z

    def generate(self, node_num, edge_index):
        z = torch.randn((n, self.hidden_dim)).to(edge_index.device)
        recon_x = self.decoder(z, edge_index)
        return recon_x

