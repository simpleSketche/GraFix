

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
        cur_path = cur_path[:-1]
        return cur_path

    def parse_path(self):
        root_path = self.get_cur_path()
        root_path.append('data\\edges')
        new_path = '\\'.join(root_path)
        is_path_exist = os.path.exists(new_path)
        if(is_path_exist==False):
            os.makedirs(new_path)
        return new_path
    
    def parse_json_path(self):
        json_path = self.get_cur_path()
        json_path.append('data\\edges\\{}.json'.format(int(self.option)))
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
