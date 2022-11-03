

import rhinoinside
import Rhino.Geometry as rg
import random
import math
import box

class Stack_bricks(object):
    def __init__(self, brick_data, rules, target_n, start_plane, option, use_rule_seq, rule_seq):
        self.brick_data = brick_data
        self.rules = rules
        self.target_n = target_n
        self.start_plane = start_plane
        self.bricks = []
        self.connected_ports = []
        self.connected_bricks = []
        self.connected_ports_name = []
        self.grow_rule_seq = []
        self.brick = box.Brick
        self.use_rule_seq = use_rule_seq
        self.rule_seq = rule_seq
        random.seed(option)
    
    def orient_brick(self, cur_port, target_port):
        # I don't have time to clean this, but I think there must be a simpler implementation
        # of this logic. The general idea here is to orient the block based on which port it connects to.
        if cur_port == target_port:
            rotation = math.pi
        elif cur_port == "top" and target_port != "bottom":
            if target_port == "right":
                rotation = math.pi/2
            elif target_port == "left":
                rotation = math.pi/2 * -1
        elif cur_port == "bottom" and target_port != "top":
            if target_port == "right":
                rotation = math.pi/2 * -1
            elif target_port == "left":
                rotation = math.pi/2
        elif cur_port == "right" and target_port != "left":
            if target_port == "top":
                rotation = math.pi/2 * -1
            elif target_port == "bottom":
                rotation = math.pi/2
        elif cur_port == "left" and target_port != "right":
            if target_port == "top":
                rotation = math.pi/2
            elif target_port == "bottom":
                rotation = math.pi/2 * -1
        else:
            rotation = 0
        return rotation

    def stack_brick(self, cur_brick_data, target_brick, cur_port, target_port, id):
        # super messy here. The idea is to connect the brick after it's oriented correctly.
        cur_brick = self.brick(cur_brick_data["depth"], cur_brick_data["width"], cur_brick_data["type"], id)
        rotation = self.orient_brick(cur_port, target_port)
        target_brick.ports[target_port]["plane"].Rotate(rotation, rg.Vector3d.ZAxis)
        target_port_plane = target_brick.ports[target_port]["plane"]
        cur_port_plane = cur_brick.ports[cur_port]["plane"]
        t = rg.Transform.PlaneToPlane(cur_port_plane, target_port_plane)
        self.connected_ports.append(cur_port_plane)
        self.connected_bricks.append(target_brick.footprint)
        self.connected_ports_name.append(str(target_brick) + " " + target_port)
        cur_brick.move(t)
        cur_brick.update_ports(cur_port, target_port, target_brick)
        cur_brick.rotation = rotation
        return cur_brick
    
    def add_brick(self, new_brick):
        self.bricks.append(new_brick)
    
    def all_open_ports(self):
        open_ports = []
        for brick in self.bricks:
            for port in brick.ports:
                if brick.ports[port]["neighbor"] == None:
                    open_ports.append([brick, port])
        target_brick, target_port = random.choice(open_ports)
        return target_brick, target_port
    
    def test_collision(self, cur_brick, all_bricks):
        isCollision = False
        for brick in all_bricks:
            intersection = rg.Intersect.Intersection.CurveCurve(cur_brick.footprint.ToNurbsCurve(), brick.footprint.ToNurbsCurve(), 0.00000001, 0.00000001)
            if(len(intersection) > 1):
                isCollision = True
                break
        return isCollision
    
    def generate(self):
        # the main loop that generates the brick mass. Two rules here, you either give it a pre-made detemrinistic rule ids.
        # or adjust the target_n to randomly generate one.
        start_brick_data = random.choice(self.brick_data,)
        start_brick = self.brick(start_brick_data["depth"], start_brick_data["width"], start_brick_data["type"], 0)
        self.add_brick(start_brick)
        start_brick.rotation = 0
        
        if self.use_rule_seq:
            target_num = len(self.rule_seq)
        else:
            target_num = self.target_n
        
        cur_n = 0
        
        while cur_n < target_num:
            target_brick, target_port = self.all_open_ports()
            cur_brick_data = random.choice(self.brick_data)
            if self.use_rule_seq:
                sel_rule_idx = int(self.rule_seq[cur_n])
                cur_port = self.rules[sel_rule_idx]
            else:
                cur_port = random.choice(self.rules)
                self.grow_rule_seq.append(cur_port)
            new_brick_id = cur_n + 1
            cur_brick = self.stack_brick(cur_brick_data, target_brick, cur_port, target_port, new_brick_id)
            is_collision = self.test_collision(cur_brick, self.bricks)
            cur_brick.intersection = is_collision
            self.add_brick(cur_brick)
            cur_n += 1




