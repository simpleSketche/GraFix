from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from datetime import datetime
import os
import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from dataset import *
from model import *
from utils import *

d = RoomAsNodeRegressionDataset(root='sample_test_data', data_num=1)


model_args = {"input_dim": 8+GRAPH_BASED_FEATURE_DIM, "hidden_dim": 256, "output_dim": 8,
   			 	  "num_layers": 1, "message_passing": 'GAT'}
model = globals()['GNN'](**model_args).cuda()
model.load_state_dict(torch.load('Results/RoomAsNodeRegression/2022_11_06__22_25_18/model.pt'))

model.eval()

save_path = 'sample_test_data/pred_1.pt'
graph_i = d[0].cuda()
pred_i = model(graph_i.x, graph_i.edge_index)
output_graph_regression(save_path, graph_i, pred_i)

from OsUtils import make_di_path, wipe_dir,save_pickle
def to_csv(arr, dir_name):
    np.savetxt(dir_name, arr, delimiter=",")
    
graph_fname_list=['sample_test_data/pred_1.pt']
save_dir = 'sample_test_data/pred_csv_test'
make_di_path(save_dir)
make_di_path(save_dir+'/nodes_in')
make_di_path(save_dir+'/nodes_out')
make_di_path(save_dir+'/move_pred')
make_di_path(save_dir+'/nodes_out_gt')
make_di_path(save_dir+'/edges')

count=0
with torch.no_grad():
    for fname_i in graph_fname_list:
        graph_i = torch.load(fname_i)
        dir_name_x = save_dir+'/nodes_in'+'/'+str(count)+'.csv'
        arr_x=graph_i['x'].detach().cpu().numpy()
        to_csv(arr_x, dir_name_x)

        dir_name_edge = save_dir+'/edges'+'/'+str(count)+'.csv'
        arr_edge = graph_i['edge_index'].detach().cpu().numpy()
        to_csv(arr_edge, dir_name_edge)
        
        arr_y=graph_i['y'].detach().cpu().numpy()
        dir_name_y = save_dir+'/nodes_out'+'/'+str(count)+'.csv'
        dir_name_y_gt = save_dir+'/nodes_out_gt'+'/'+str(count)+'.csv'
        arr_mov_gt = arr_x+ arr_y
        to_csv(arr_mov_gt, dir_name_y_gt)
        
        arr_mov = arr_x + graph_i['vertices_movement_prediction'].detach().cpu().numpy()
        to_csv(arr_mov, dir_name_y)
        
        dir_name_pred = save_dir+'/move_pred'+'/'+str(count)+'.csv'
        arr_del = graph_i['vertices_movement_prediction'].detach().cpu().numpy()
        to_csv(arr_del, dir_name_pred)
        # save_pickle(dict_i, 'corners_in', dir_name_x)
        count+=1
        
        print(count)

