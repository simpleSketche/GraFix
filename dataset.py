import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import os
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import networkx as nx
from copy import deepcopy


NORM_DICT = {
    "coord": 20,
    "center_distance": 17,
    "movement": 0.5,
}

MAX_ROOM_NUM = 9

FILL_VALUE_X_NO_ROOM = -1
FILL_VALUE_Y_NO_ROOM = -10000

GRAPH_BASED_FEATURE_DIM = 5


class RoomAsNodeDataset(Dataset):
    def __init__(self, root="data/room_base_graph", data_num=100):
        root = Path(root)
        self.nodes_in_folder = root / "nodes_in"
        self.nodes_out_folder = root / "nodes_out"
        self.edges_folder = root / "edges"
        self.graphs = []
        for data_name in os.listdir(self.nodes_in_folder)[:data_num]:
            graph = self._construct_graph(data_name)
            graph = self._normalize(graph)
            self.graphs.append(graph)

    
    def _construct_graph(self, data_name: str):
        nodes_in_data = json.loads((self.nodes_in_folder / data_name).read_text())
        nodes_out_data = json.loads((self.nodes_out_folder / data_name).read_text())
        edges_data = json.loads((self.edges_folder / data_name).read_text())

        # node x (nodes_in_data)
        room_number = len(nodes_in_data["corners"])
        node_vertices_feature_dim = 8
        x = torch.full((room_number, node_vertices_feature_dim + GRAPH_BASED_FEATURE_DIM), FILL_VALUE_X_NO_ROOM).to(torch.float)
        for room_index in range(room_number):
            for vertices_index, (vertices_x, vertices_y) in enumerate(nodes_in_data["corners"][room_index][:-1]):
                x[room_index, vertices_index*2] = float(vertices_x)
                x[room_index, vertices_index*2+1] = float(vertices_y)

        # edge_index (edges_data)
        edge_list = []
        for edge_pair in edges_data["adjacency"]:
            if edge_pair in edge_list: continue
            index_1, index_2 = edge_pair
            edge_list.append([index_1, index_2])
            edge_list.append([index_2, index_1])
        edge_index = torch.tensor(edge_list).T

        '''
        # edge_attr
        # box center's distance
        # box connection type, one hot for (left, right, top, left)
        edge_attr = torch.zeros((edge_index.shape[1], 5))
        for i, (index_1, index_2) in enumerate(edge_list):
            center_x_1 = torch.mean(x[index_1, [0, 2, 4, 6]])
            center_y_1 = torch.mean(x[index_1, [1, 3, 5, 7]])
            center_x_2 = torch.mean(x[index_2, [0, 2, 4, 6]])
            center_y_2 = torch.mean(x[index_2, [1, 3, 5, 7]])
            width_1 = torch.abs(x[index_1, 0] - x[index_1, 2])
            height_1 = torch.abs(x[index_1, 1] - x[index_1, 3])
            distance = ((center_x_1 - center_x_2)**2 + (center_y_1 - center_y_2)**2) ** 0.5
            edge_attr[i, 0] = distance
            if center_x_2 < (center_x_1 - width_1/2):
                edge_attr[i, 1] = 1
            elif center_x_2 > (center_x_1 + width_1/2):
                edge_attr[i, 2] = 1
            elif center_y_2 > (center_y_1 + height_1/2):
                edge_attr[i, 3] = 1
            elif center_y_2 < (center_y_1 - height_1/2):
                edge_attr[i, 4] = 1
            else:
                raise ValueError("Connection does not exist!")
        '''


        # node y (nodes_out_data)
        y = torch.full((room_number, node_vertices_feature_dim), FILL_VALUE_Y_NO_ROOM).to(torch.float)
        for room_index in range(room_number):
            for vertices_index, (vertices_x, vertices_y) in enumerate(nodes_out_data["corners"][room_index][:-1]):
                y[room_index, vertices_index*2] = float(vertices_x) - x[room_index, vertices_index*2]
                y[room_index, vertices_index*2+1] = float(vertices_y) - x[room_index, vertices_index*2+1]


        # Add traditional node features as input
        # first construct graph with networkx
        G = nx.Graph()
        for i in range(room_number):
            G.add_node(i)
        for edge_i, edge_j in edge_list:
            G.add_edge(edge_i, edge_j)
        # node degree
        degree = [pair[1] for pair in G.degree()]
        eigenvector_centrality = list(nx.eigenvector_centrality(G, max_iter=1000).values())
        # closness centrality
        closeness_centrality = list(nx.closeness_centrality(G).values())
        # betweenness_centrality
        betweenness_centrality = list(nx.betweenness_centrality(G).values())
        # clustering coefficient
        clustering_coefficient = list(nx.clustering(G).values())
        # assign to node feature
        x[:, node_vertices_feature_dim+0] = torch.tensor(degree)
        x[:, node_vertices_feature_dim+1] = torch.tensor(eigenvector_centrality)
        x[:, node_vertices_feature_dim+2] = torch.tensor(closeness_centrality)
        x[:, node_vertices_feature_dim+3] = torch.tensor(betweenness_centrality)
        x[:, node_vertices_feature_dim+4] = torch.tensor(clustering_coefficient)

        # graph
        graph = Data(x=x, edge_index=edge_index, y=y, vertices_movement_prediction=None, original_x=None, original_y=None)
        return graph

    def _normalize(self, graph):
        graph.original_x = deepcopy(graph.x[:, 0:8])
        graph.original_y = deepcopy(graph.y)
        graph.x[:, 0:8] /= NORM_DICT["coord"]
        graph.y /= NORM_DICT["movement"]
        return graph
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, i):
        return self.graphs[i]








class RoomAsNodeSanityCheckDataset(Dataset):
    def __init__(self, root="data/room_base_graph", data_num=100):
        root = Path(root)
        self.nodes_in_folder = root / "nodes_in"
        self.nodes_out_folder = root / "nodes_out"
        self.edges_folder = root / "edges"
        self.graphs = []
        for data_name in os.listdir(self.nodes_in_folder)[:data_num]:
            graph = self._construct_graph(data_name)
            graph = self._normalize(graph)
            self.graphs.append(graph)
    
    def _construct_graph(self, data_name: str):
        nodes_in_data = json.loads((self.nodes_in_folder / data_name).read_text())
        nodes_out_data = json.loads((self.nodes_out_folder / data_name).read_text())
        edges_data = json.loads((self.edges_folder / data_name).read_text())

        # node x (nodes_in_data)
        room_number = len(nodes_in_data["corners"])
        node_vertices_feature_dim = 8
        x = torch.full((room_number, node_vertices_feature_dim + GRAPH_BASED_FEATURE_DIM), FILL_VALUE_X_NO_ROOM).to(torch.float)
        for room_index in range(room_number):
            for vertices_index, (vertices_x, vertices_y) in enumerate(nodes_in_data["corners"][room_index][:-1]):
                x[room_index, vertices_index*2] = float(vertices_x)
                x[room_index, vertices_index*2+1] = float(vertices_y)

        # edge_index (edges_data)
        edge_list = []
        for edge_pair in edges_data["adjacency"]:
            if edge_pair in edge_list: continue
            index_1, index_2 = edge_pair
            edge_list.append([index_1, index_2])
            edge_list.append([index_2, index_1])
        edge_index = torch.tensor(edge_list).T

        # node y (nodes_out_data)
        y = torch.full((room_number, node_vertices_feature_dim), 1).to(torch.float)


        # Add traditional node features as input
        # first construct graph with networkx
        G = nx.Graph()
        for i in range(room_number):
            G.add_node(i)
        for edge_i, edge_j in edge_list:
            G.add_edge(edge_i, edge_j)
        # node degree
        degree = [pair[1] for pair in G.degree()]
        eigenvector_centrality = list(nx.eigenvector_centrality(G, max_iter=1000).values())
        # closness centrality
        closeness_centrality = list(nx.closeness_centrality(G).values())
        # betweenness_centrality
        betweenness_centrality = list(nx.betweenness_centrality(G).values())
        # clustering coefficient
        clustering_coefficient = list(nx.clustering(G).values())
        # assign to node feature
        x[:, node_vertices_feature_dim+0] = torch.tensor(degree)
        x[:, node_vertices_feature_dim+1] = torch.tensor(eigenvector_centrality)
        x[:, node_vertices_feature_dim+2] = torch.tensor(closeness_centrality)
        x[:, node_vertices_feature_dim+3] = torch.tensor(betweenness_centrality)
        x[:, node_vertices_feature_dim+4] = torch.tensor(clustering_coefficient)

        # graph
        graph = Data(x=x, edge_index=edge_index, y=y, vertices_movement_prediction=None, original_x=None, original_y=None)
        return graph

    def _normalize(self, graph):
        graph.original_x = deepcopy(graph.x[:, 0:8])
        graph.original_y = deepcopy(graph.y)
        graph.x[:, 0:8] /= NORM_DICT["coord"]
        graph.y /= 1
        return graph
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, i):
        return self.graphs[i]




def output_graph_RoomAsNode(save_path, norm_graph, prediction):
        x = norm_graph.original_x
        y = norm_graph.original_y
        edge_index = norm_graph.edge_index
        vertices_movement_prediction = prediction * NORM_DICT["movement"]
        graph = Data(x=x, edge_index=edge_index, y=y, vertices_movement_prediction=vertices_movement_prediction)
        torch.save(graph, save_path)




if __name__ == "__main__":
    dset = RoomAsNodeDataset(data_num=1000)
    print(dset)

