import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import networkx as nx


NORM_DICT = {
    "length": 10,
    "location": 10,
}



class BoxDataset(Dataset):
    def __init__(self, root="data", data_num=100):
        root = Path(root)
        self.graphs = []
        for i in range(1, data_num+1):
            graph = self._construct_graph(root, i)
            if graph != None:
                graph = self._normalize(graph)
                self.graphs.append(graph)
    

    def _construct_graph(self, root: Path, i: int):
        node_data = json.loads((root / "nodes" / f"{i}.json").read_text())
        edge_data = json.loads((root / "edges" / f"{i}.json").read_text())

        # check if intersection
        intersections = node_data["intersection"]
        if sum(intersections) > 0:
            return None

        # node x
        node_number = len(node_data["node_id"]) - 1
        node_geometric_feature_dim = 5
        traditional_node_feature_dim = 5
        x = torch.zeros(node_number, node_geometric_feature_dim + traditional_node_feature_dim)
        node_location = {}
        for node_index in range(1, node_number+1):
            x[node_index-1, 0] = node_data["width"][node_index]
            x[node_index-1, 1] = node_data["length"][node_index]
            x[node_index-1, 2] = node_data["rotation"][node_index]
            x[node_index-1, 3] = node_data["location"][node_index][0]
            x[node_index-1, 4] = node_data["location"][node_index][1]

        # edge_index
        edge_list = []
        for edge_pair in edge_data["adjacency"]:
            if -1 in edge_pair: continue
            if edge_pair in edge_list: continue
            index_1, index_2 = edge_pair
            edge_list.append([index_1, index_2])
            edge_list.append([index_2, index_1])
        edge_index = torch.tensor(edge_list).T

        # Add traditional node features as input
        # first construct graph with networkx
        G = nx.Graph()
        for i in range(node_number):
            G.add_node(i)
        for edge_i, edge_j in edge_list:
            G.add_edge(edge_i, edge_j)
        # node degree
        degree = [pair[1] for pair in G.degree()]
        # eigenvector centrality
        eigenvector_centrality = list(nx.eigenvector_centrality(G).values())
        # closness centrality
        closeness_centrality = list(nx.closeness_centrality(G).values())
        # betweenness_centrality
        betweenness_centrality = list(nx.betweenness_centrality(G).values())
        # clustering coefficient
        clustering_coefficient = list(nx.clustering(G).values())
        # assign to node feature
        x[:, 5] = torch.tensor(degree)
        x[:, 6] = torch.tensor(eigenvector_centrality)
        x[:, 7] = torch.tensor(closeness_centrality)
        x[:, 8] = torch.tensor(betweenness_centrality)
        x[:, 9] = torch.tensor(clustering_coefficient)

        # graph
        graph = Data(x=x, edge_index=edge_index)
        return graph

    def _normalize(self, graph):
        graph.x[:, 0:2] /= NORM_DICT["length"]
        graph.x[:, 3:5] /= NORM_DICT["location"]
        return graph
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, i):
        return self.graphs[i]




if __name__ == "__main__":
    dset = BoxDataset(data_num=1)
    print(dset)

