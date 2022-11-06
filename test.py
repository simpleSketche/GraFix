import torch

path = "Results/RoomAsNodeClassification/2022_11_06__18_08_03/Prediction_valid/pred_0.pt"
graph = torch.load(path)

print(graph.x)
print(graph.edge_index)
print(graph.y)
print(graph.vertices_movement_prediction)