


import rhinoinside
import Rhino
import Rhino.Geometry as rg
import math
import json
import os

class Node_json():
    def __init__(self, boxes, site_width, site_length, siteLoc, option):
        self.boxes = boxes
        self.site_width = site_width
        self.site_length = site_length
        self.siteLoc = siteLoc
        self.option = option
    
    def get_cur_path(self):
        cur_path = os.getcwd().split('\\')
        cur_path = cur_path[:-1]
        return cur_path

    def parse_path(self):
        root_path = self.get_cur_path()
        root_path.append('data\\room_base_graph\\nodes_in')
        new_path = '\\'.join(root_path)
        is_path_exist = os.path.exists(new_path)
        if(is_path_exist==False):
            os.makedirs(new_path)
        return new_path
    
    def parse_json_path(self):
        json_path = self.get_cur_path()
        json_path.append('data\\room_base_graph\\nodes_in\\{}.json'.format(int(self.option)))
        new_json_path = '\\'.join(json_path)
        return new_json_path
    
    def parse_path_out(self):
        root_path = self.get_cur_path()
        root_path.append('data\\room_base_graph\\nodes_out')
        new_path = '\\'.join(root_path)
        is_path_exist = os.path.exists(new_path)
        if(is_path_exist==False):
            os.makedirs(new_path)
        return new_path
    
    def parse_json_path_out(self):
        json_path = self.get_cur_path()
        json_path.append('data\\room_base_graph\\nodes_out\\{}.json'.format(int(self.option)))
        new_json_path = '\\'.join(json_path)
        return new_json_path

    def create_json_data(self):
        output_nodes_data = {"corners":[], "node_id":[]}
        # assign the rest of nodes
        for node in self.boxes:
            corners = self.convert_footprint_to_pts(node.footprint)
            output_nodes_data["corners"].append(corners)
            output_nodes_data["node_id"].append(node.id)
        return output_nodes_data
    
    def save_json_data(self):
        cur_path = self.parse_json_path()
        node_data = self.create_json_data()
        with open(cur_path, 'w') as f:
            f.write(json.dumps(node_data))

    def save_json_data_out(self):
        cur_path = self.parse_json_path_out()
        node_data = self.create_json_data()
        with open(cur_path, 'w') as f:
            f.write(json.dumps(node_data))

    def convert_footprint_to_pts(self, polyline):
        pts = []
        for pt in polyline:
            x = pt.X
            y = pt.Y
            pts.append([x, y])
        return pts