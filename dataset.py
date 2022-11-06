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


class RoomAsNodeRegressionDataset(Dataset):
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





class RoomAsNodeClassificationDataset(Dataset):
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
        x = torch.zeros((room_number, node_vertices_feature_dim + GRAPH_BASED_FEATURE_DIM)).to(torch.float)
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
        vertices_num_for_box = 4
        displacement_threshold = 1e-4
        y = torch.zeros((room_number, vertices_num_for_box)).to(torch.float)
        for room_index in range(room_number):
            for vertices_index, (vertices_x, vertices_y) in enumerate(nodes_out_data["corners"][room_index][:-1]):
                disp_x = abs(float(vertices_x) - x[room_index, vertices_index*2])
                disp_y = abs(float(vertices_y) - x[room_index, vertices_index*2+1])
                if disp_x > displacement_threshold or disp_y > displacement_threshold:
                    y[room_index, vertices_index] = 1
                else:
                    y[room_index, vertices_index] = 0

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
        return graph
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, i):
        return self.graphs[i]



def output_graph_regression(save_path, norm_graph, prediction):
        x = norm_graph.original_x
        y = norm_graph.original_y
        edge_index = norm_graph.edge_index
        vertices_movement_prediction = prediction * NORM_DICT["movement"]
        graph = Data(x=x, edge_index=edge_index, y=y, vertices_movement_prediction=vertices_movement_prediction)
        torch.save(graph, save_path)


def output_graph_classification(save_path, norm_graph, prediction):
        x = norm_graph.original_x
        y = norm_graph.original_y
        edge_index = norm_graph.edge_index
        vertices_movement_prediction = torch.round(prediction)
        graph = Data(x=x, edge_index=edge_index, y=y, vertices_movement_prediction=vertices_movement_prediction)
        torch.save(graph, save_path)




if __name__ == "__main__":
    dset = RoomAsNodeDataset(data_num=1000)
    print(dset)

