import torch

def binary_classification_accracy(pred, y):
    pred = torch.round(pred)
    acc = torch.sum(pred == y) / torch.numel(y)
    return acc