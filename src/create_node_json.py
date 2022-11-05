


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
        root_path.append('data\\legacy\\nodes')
        new_path = '\\'.join(root_path)
        is_path_exist = os.path.exists(new_path)
        if(is_path_exist==False):
            os.makedirs(new_path)
        return new_path
    
    def parse_json_path(self):
        json_path = self.get_cur_path()
        json_path.append('data\\legacy\\nodes\\{}.json'.format(int(self.option)))
        new_json_path = '\\'.join(json_path)
        return new_json_path

    def create_json_data(self):
        output_nodes_data = {"node_type":["S"], "width":[round(self.site_width,5)], "length":[round(self.site_length,5)], "location":[[round(self.siteLoc.X,5), round(self.siteLoc.Y,5)]], "rotation":[0], "intersection":[False], "node_id":[-1]}
        # assign the rest of nodes
        for node in self.boxes:
            center_location = rg.AreaMassProperties.Compute(node.footprint.ToNurbsCurve()).Centroid
            rotation_degree = round(node.rotation * 57.2958) # convert to degrees
            output_nodes_data["node_type"].append(node.type)
            output_nodes_data["width"].append(round(node.width,5))
            output_nodes_data["length"].append(round(node.depth,5))
            output_nodes_data["location"].append([round(center_location.X,5), round(center_location.Y,5)])
            output_nodes_data["intersection"].append(node.intersection)
            output_nodes_data["node_id"].append(node.id)
            rotation = output_nodes_data["rotation"]
            
            if(rotation_degree == 0 or rotation_degree == 180):
                rotation.append(0) # if the box is not rotated
            else:
                rotation.append(1) # if the box is rotated
        return output_nodes_data
    
    def save_json_data(self):
        cur_path = self.parse_json_path()
        node_data = self.create_json_data()
        with open(cur_path, 'w') as f:
            f.write(json.dumps(node_data))