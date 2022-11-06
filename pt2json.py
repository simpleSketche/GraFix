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

results_dir ='C:/Users/SSajedi/Desktop/GitHub/BoxStacker/Results/2022_11_06__10_59_04/Prediction'

graph_fname_list = glob.glob(results_dir+'/*.pt')

def np2list(arr):
    list_arr=[]
    for i in range(arr.shape[0]):
        list_arr.append(list(arr[i]))
    return list_arr

def to_json(dict_i,dir_name):
    with open(dir_name, "w") as outfile:
        json.dump(dict_i, outfile)    
    

count = 0

save_dir = 'pred_pkl'
make_di_path(save_dir)
# make_di_path(save_dir+'/nodes_in')
# make_di_path(save_dir+'/nodes_out')
# make_di_path(save_dir+'/edges')

with torch.no_grad():
    for fname_i in graph_fname_list:
        graph_i = torch.load(fname_i)
        dir_name_x = save_dir+'/'+str(count)
        
        dict_i={}
        arr_x=graph_i['x'].detach().cpu().numpy()
        dict_i['corners_in'] = np2list(arr_x)

        arr_edge = graph_i['edge_index'].detach().cpu().numpy()
        dict_i['edge_index']=np2list(arr_edge)
        dict_i['corners_out']=graph_i['y'].detach().cpu().numpy()
        dict_i['vertices_movement_prediction']=graph_i['vertices_movement_prediction'].detach().cpu().numpy()
        save_pickle(dict_i, dir_name_x)
        count+=1
        print(count)