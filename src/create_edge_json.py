"""Provides a scripting component.
    Inputs:
        x: The x script variable
        y: The y script variable
    Output:
        a: The a output variable"""

__author__ = "YYang"
__version__ = "2022.10.31"

import rhinoinside
import Rhino
import Rhino.Geometry as rg
import math
import json
import os


class Edge_json():
    def __init__(self, boxes, siteWidth, siteLength, siteLoc, option):
        self.boxes = boxes
        self.siteWidth = siteWidth
        self.siteLength = siteLength
        self.siteLoc = siteLoc
        self.option = option
    
    def get_cur_path(self):
        cur_path = os.getcwd().split('\\')
        return cur_path

    def parse_path(self):
        root_path = self.get_cur_path()
        root_path.append('training\\trial1\\edges')
        new_path = '\\'.join(root_path)
        is_path_exist = os.path.exists(new_path)
        if(is_path_exist==False):
            os.makedirs(new_path)
        return new_path
    
    def parse_json_path(self):
        json_path = self.get_cur_path()
        json_path.append('training\\trial1\\edges\\{}.json'.format(int(self.option)))
        new_json_path = '\\'.join(json_path)
        return new_json_path

    def is_edge(self, cur_node, other_node):
        isEdge = False
        intersection = rg.Intersect.Intersection.CurveCurve(cur_node.footprint.ToNurbsCurve(), other_node.footprint.ToNurbsCurve(), 0.00000001, 0.00000001)
        if(len(intersection) == 1):
            isEdge = True
        return isEdge

    def create_json_data(self):
        output_edges_data = {"adjacency": []}
        for box in self.boxes:
            # by default, the site is a virtual node that is the parent of all nodes
            # and the hard-fixed id for site is always -1
            output_edges_data["adjacency"].append([-1, box.id])
            for otherbox in self.boxes:
                if(box.id != otherbox.id):
                    isCreateEdge = self.is_edge(box, otherbox)
                    if(isCreateEdge):
                        cur_edge = [box.id, otherbox.id]
                        output_edges_data["adjacency"].append(cur_edge)
        return output_edges_data
    
    def save_json_data(self):
        cur_path = self.parse_json_path()
        edge_data = self.create_json_data()
        with open(cur_path, 'w') as f:
            f.write(json.dumps(edge_data))


# cur_path.append('training\\trial1\\nodes\\{}.json'.format(int(option)))

# cur_path = "\\".join(cur_path)
# print (cur_path)

# site_length = abs(siteLength[0] - siteLength[1])
# site_width = abs(siteWidth[0] - siteWidth[1])

# """
# The root node of each design option graph should be the site,
# and each node should connected to the site
# """

# # output data format
# # {"node_type":"", "width":0, "length":0, "location":[x,y], "rotation":0,"intersection":[False], "node_id":-1}

# # assign site first, S -> site
# output_nodes_data = {"node_type":["S"], "width":[round(site_width,5)], "length":[round(site_length,5)], "location":[[round(siteLoc.X,5), round(siteLoc.Y,5)]], "rotation":[0], "intersection":[False], "node_id":[-1]}

# # assign the rest of nodes
# for node in boxes:
#     center_location = rg.AreaMassProperties.Compute(node.footprint.ToNurbsCurve()).Centroid
#     rotation_degree = round(node.rotation * 57.2958) # convert to degrees
#     output_nodes_data["node_type"].append(node.type)
#     output_nodes_data["width"].append(round(node.width,5))
#     output_nodes_data["length"].append(round(node.depth,5))
#     output_nodes_data["location"].append([round(center_location.X,5), round(center_location.Y,5)])
#     output_nodes_data["intersection"].append(node.intersection)
#     output_nodes_data["node_id"].append(node.id)
#     rotation = output_nodes_data["rotation"]
    
#     if(rotation_degree == 0 or rotation_degree == 180):
#         rotation.append(0) # if the box is not rotated
#     else:
#         rotation.append(1) # if the box is rotated

# result = output_nodes_data

# with open(cur_path, 'w') as f:
#     f.write(json.dumps(result))