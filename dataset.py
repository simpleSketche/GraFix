import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import networkx as nx


NORM_DICT = {
    "length": 10
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
        x = torch.zeros(node_number, 2+5)  # width, length
        node_location = {}
        for node_index in range(1, node_number+1):
            if node_data["rotation"][node_index] == 0:
                x[node_index-1, 0] = node_data["width"][node_index]
                x[node_index-1, 1] = node_data["length"][node_index]
            else:
                x[node_index-1, 0] = node_data["length"][node_index]
                x[node_index-1, 1] = node_data["width"][node_index]
            
            # location
            node_location[node_index-1] = node_data["location"][node_index]

        # connection type, 0: vertical connection, 1: horizontal connection
        def get_edge_type(index_1, index_2):
            loc1_x, loc1_y = node_location[index_1]
            width_1 = x[index_1][0]
            height_2 = x[index_1][1]
            loc2_x, loc2_y = node_location[index_2]
            if loc2_x < loc1_x - width_1/2 or loc2_x > loc1_x + width_1:
                return 1.0
            return 0.0

        # edge_index
        edge_list = []
        edge_type = []     # connection type, not defined yet, need to consider locations to define
        for edge_pair in edge_data["adjacency"]:
            if -1 in edge_pair: continue
            if edge_pair in edge_list: continue
            index_1, index_2 = edge_pair
            edge_list.append([index_1, index_2])
            edge_list.append([index_2, index_1])
            # edge type
            e_type = get_edge_type(index_1, index_2)
            edge_type.append(e_type)
            edge_type.append(e_type)

        edge_index = torch.tensor(edge_list).T
        edge_type = torch.tensor(edge_type)

        # add traditional node features as input
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
        x[:, 2] = torch.tensor(degree)
        x[:, 3] = torch.tensor(eigenvector_centrality)
        x[:, 4] = torch.tensor(closeness_centrality)
        x[:, 5] = torch.tensor(betweenness_centrality)
        x[:, 6] = torch.tensor(clustering_coefficient)

        # graph
        graph = Data(x=x, edge_index=edge_index, edge_type=edge_type)
        return graph

    def _normalize(self, graph):
        graph.x[:, 0:2] /= NORM_DICT["length"]
        return graph
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, i):
        return self.graphs[i]




if __name__ == "__main__":
    dset = BoxDataset(data_num=1)
    print(dset)

