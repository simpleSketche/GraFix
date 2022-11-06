import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataset import *


# accuracy

def binary_classification_accracy(pred, y):
    pred = torch.round(pred)
    acc = torch.sum(pred == y) / torch.numel(y)
    return acc

def R2_score(pred, y):
    # print("in r2 score: y")
    # print(y)
    # print()
    y_bar = torch.sum(torch.mean(y))
    SS_tot = torch.sum((y - y_bar)**2)
    SS_res = torch.sum((y - pred)**2)
    R2 = 1 - SS_res / SS_tot
    return R2







