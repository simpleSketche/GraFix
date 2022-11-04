import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling


NORM_DICT = {
    "length": 10
}



class BoxDataset(Dataset):
    def __init__(self, root="data", data_num=100):
        root = Path(root)
        self.graphs = []
        for i in range(1, data_num+1):
            graph = self._construct_graph(root, i)
            graph = self._normalize(graph)
            self.graphs.append(graph)
    
    def _construct_graph(self, root: Path, i: int):
        node_data = json.loads((root / "nodes" / f"{i}.json").read_text())
        edge_data = json.loads((root / "edges" / f"{i}.json").read_text())

        # node x
        node_number = len(node_data["node_id"]) - 1
        x = torch.zeros(node_number, 5)  # node_type(A1, B1, C1), width, length
        for node_index in range(1, node_number+1):
            if node_data["node_type"][node_index] == "A1":
                x[node_index-1, 0] = 1
                
            elif node_data["node_type"][node_index] == "B1":
                x[node_index-1, 1] = 1
            elif node_data["node_type"][node_index] == "C1":
                x[node_index-1, 2] = 1
            x[node_index-1, 3] = node_data["width"][node_index]
            x[node_index-1, 4] = node_data["length"][node_index]
        
        # node y
        node_y = torch.zeros(node_number, 1)    # rotate 180 or not
        for node_index in range(1, node_number+1):
            node_y[node_index-1] = 1 if node_data["rotation"][node_index] in [1, 3] else 0

        # edge_index
        edge_list = []
        edge_y = [0]     # connection type, not defined yet, need to consider locations to define
        for edge_pair in edge_data["adjacency"]:
            if -1 in edge_pair: continue
            if edge_pair in edge_list: continue
            index_1, index_2 = edge_pair
            edge_list.append([index_1, index_2])
            edge_list.append([index_2, index_1])
        edge_index = torch.tensor(edge_list).T
        edge_y = torch.tensor(edge_y)

        # negative edge_index
        num_neg_samples = node_number ** 2 - edge_index.size(1)
        all_neg_edge_index = negative_sampling(edge_index, num_neg_samples=num_neg_samples)
        neg_edge_index = []
        for edge_pair in all_neg_edge_index.T:
            if edge_pair.tolist() in edge_list: continue
            if edge_pair.tolist()[::-1] in edge_list: continue
            if edge_pair[0] == edge_pair[1]: continue
            neg_edge_index.append(edge_pair.tolist())
        neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long).T

        # graph
        graph = Data(x=x, edge_index=edge_index, neg_edge_index=neg_edge_index, node_y=node_y, edge_y=edge_y)
        return graph

    def _normalize(self, graph):
        graph.x[:, 3:5] /= NORM_DICT["length"]
        return graph
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, i):
        return self.graphs[i]




if __name__ == "__main__":
    dset = BoxDataset(data_num=1)
    print(dset)

