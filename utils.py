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
    y_bar = torch.sum(torch.mean(y))
    SS_tot = torch.sum((y - y_bar)**2)
    SS_res = torch.sum((y - pred)**2)
    R2 = 1 - SS_res / SS_tot
    return R2



# visualize

# def edge_index_to_list(edge_index, increased=True):
#     max_index = torch.max(edge_index)
#     edge_list = {index: [] for index in range(max_index)}
#     for idx_1, idx_2 in edge_index.T:
#         if increased:
#             if idx_1 


BOX_SHAPE = {
    # [width, height]
    "A1": [8, 6],
    "B1": [8, 15],
    "C1": [16, 6]
}

BOX_COLOR = {
    "A1": "pink",
    "B1": "mediumorchid",
    "C1": "orange"
}



def visualize_generation(recon_x, edge_index, box_types=None, save_path=None):
    box_num = recon_x.size(0)
    box_types = ["A1" for _ in range(box_num)] if box_types is None else box_types
    rotation = torch.round(recon_x[:, 0])
    center_x = recon_x[:, 1].cpu().numpy() * NORM_DICT["location"]
    center_y = recon_x[:, 2].cpu().numpy() * NORM_DICT["location"]

    # plot
    fig, ax = plt.subplots()
    for box_idx in range(box_num):
        box_type = box_types[box_idx]
        width, height = BOX_SHAPE[box_type]
        if rotation[box_idx] == 1:  # is rotated
            width, height = height, width
        cen_x, cen_y = center_x[box_idx], center_y[box_idx]
        anchor_x = cen_x - width/2
        anchor_y = cen_y - height/2
        # print(f"anchor: ({anchor_x}, {anchor_y}), shape: ({width}, {height})")
        # plot rectangle
        ax.add_patch(Rectangle((anchor_x, anchor_y), width, height, edgecolor="black", facecolor=BOX_COLOR[box_type], fill=True))
    
    plt.xlim([-20, 30])
    plt.ylim([-20, 30])
    plt.savefig(save_path)
    plt.close()
    # plt.show()



