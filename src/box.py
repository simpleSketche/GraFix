

import rhinoinside
rhinoinside.load()
import System
import Rhino
import Rhino.Geometry as rg
import random


# This class is where all your fundamental bricks are originated from.

class Brick(object):
    
    def __init__(self, depth, width, type, id):
        self.depth = depth
        self.width = width
        self.type = type
        self.id = id
        self.origin = self.create_origin()
        self.ports = self.create_port_dict()
        self.footprint = self.display()
        self.rotation = 0
        self.intersection = False
    
    def __repr__(self):
        return "Brick_" + str(self.id)
    
    def create_origin(self):
        return rg.Plane.WorldXY
    
    def create_port_dict(self):
        ports = {}
        port_id = ["top", "left", "right", "bottom"]
        for id in port_id:
            ports[id]= {
            "plane": self.get_port_plane(id),
            "neighbor": None
            }
        return ports
    
    def get_port_plane(self, port_id):
        if port_id == "top":
            og_plane = self.origin.Clone()
            offset_y = og_plane.YAxis * (self.depth)
            offset_x = og_plane.XAxis * (self.width / 2)
            og_plane.Translate(offset_y + offset_x)
        elif port_id == "left":
            og_plane = self.origin.Clone()
            offset_y = og_plane.YAxis * (self.depth / 2)
            og_plane.Translate(offset_y)
        elif port_id == "right":
            og_plane = self.origin.Clone()
            offset_y = og_plane.YAxis * (self.depth / 2)
            offset_x = og_plane.XAxis * (self.width)
            og_plane.Translate(offset_y + offset_x)
        elif port_id == "bottom":
            og_plane = self.origin.Clone()
            offset = og_plane.XAxis * (self.width / 2)
            og_plane.Translate(offset)
        
        return og_plane
    
    def update_ports(self,this_port, other_port, other_brick):
        self.ports[this_port]["neighbor"] = other_brick
        other_brick.ports[other_port]["neighbor"] = self
    
    
    def move(self, t):
        self.origin.Transform(t)
        self.footprint.Transform(t)
        for port in self.ports:
            self.ports[port]["plane"].Transform(t)
    
    def move_random_corners(self):
        num_pts_move = random.randint(1,len(self.footprint)-2)
        for i in range(1, num_pts_move):
            randomX = round(random.uniform(-1,1), 5)
            randomY = round(random.uniform(-1,1), 5)
            self.footprint[i] = rg.Point3d(self.footprint[i].X + randomX, self.footprint[i].Y + randomY, 0)

    def display(self):
        # this function get the footprint 2d display
        pt1 = self.origin.Origin
        pt2 = rg.Point3d(self.width, self.origin.Origin.Y, 0)
        pt3 = rg.Point3d(self.width, self.depth, 0)
        pt4 = rg.Point3d(self.origin.Origin.X, self.depth, 0)
        footprint = rg.Polyline()
        footprint.Add(pt1)
        footprint.Add(pt2)
        footprint.Add(pt3)
        footprint.Add(pt4)
        footprint.Add(pt1)
        return footprint
        
