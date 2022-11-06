# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 23:05:19 2022

@author: SSajedi
"""

import torch
import glob
import json 
from OsUtils import make_di_path, wipe_dir,save_pickle
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

results_dir = 'C:/Users/SSajedi/Desktop/GitHub/BoxStacker/Results/RoomAsNode/2022_11_06__16_47_15/Prediction_valid'

graph_fname_list = glob.glob(results_dir+'/*.pt')

def np2list(arr):
    list_arr=[]
    for i in range(arr.shape[0]):
        list_arr.append(list(arr[i]))
    return list_arr

def to_csv(arr, dir_name):
    np.savetxt(dir_name, arr, delimiter=",")
    

count = 0

save_dir = 'pred_csv'
make_di_path(save_dir)
make_di_path(save_dir+'/nodes_in')
make_di_path(save_dir+'/nodes_out')
make_di_path(save_dir+'/move_pred')
make_di_path(save_dir+'/nodes_out_gt')
make_di_path(save_dir+'/edges')

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