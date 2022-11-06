import rhinoinside
rhinoinside.load()
import create_node_json_v2
import create_edge_json_v2
import math
import box
import generator
import rule
import Rhino.Geometry as rg
import option
import random




def generate(num_option):
    # loop 5000 data points
    for i in range(0, num_option):

        # create rules
        num_sels = 10
        option = i
        cur_rule = rule.Rule(num_sels, option)

        # get the rules
        rules = cur_rule.get_rules()
        rule_seq = cur_rule.get_rule_seq()

        # create dataset
        brick = box.Brick
        cur_generator = generator.Stack_bricks
        # brick parametric data
        brick_red = {"depth": 6, "width": 8, "type": "A1"}
        brick_green = {"depth": 15, "width": 8, "type": "B1"}
        brick_blue = {"depth": 6, "width": 16, "type": "C1"}
        brick_data = [brick_red, brick_green, brick_blue]
        target_n = math.floor(option / 1000) + 4
        start_plane = rg.Plane.WorldXY
        stack_bricks = cur_generator(
            brick_data, rules, target_n, start_plane, option, False, rule_seq)
        stack_bricks.generate()

        # All the stuff below are for visualization
        planes = [brick.origin for brick in stack_bricks.bricks]
        footprints = [brick.footprint for brick in stack_bricks.bricks]
        used_ports = stack_bricks.connected_ports
        used_bricks = stack_bricks.connected_bricks
        connected_ports_name = stack_bricks.connected_ports_name
        grow_rule_seq = stack_bricks.grow_rule_seq

        bricks = stack_bricks.bricks

        # create correct edges
        intersections = get_all_intersections(bricks)
        is_bad_data = test_bad_data(intersections)
        

        if(is_bad_data == False):
            # edges are from ground truth correct design option
            edge_json = create_edge_json_v2.Edge_json(
            bricks, 12, 12, rg.Point3d(0, 0, 0), option)
            edge_json.parse_path()
            edge_json.save_json_data()

            node_json = create_node_json_v2.Node_json(
                bricks, 12, 12, rg.Point3d(0, 0, 0), option)
            
            # store the ground truth data
            node_json.parse_path_out()
            node_json.save_json_data_out()

            # modify and intentionally make bad data by randomly
            # moving the boxes
            generate_error_data(bricks)
            node_json.parse_path()
            node_json.save_json_data()
            print("------------------------------------------")
            print("currently creating data point {}".format(i))
            print("Finished creating data point {}".format(i))
            print("------------------------------------------")

def generate_error_data(bricks):
    for brick in bricks:
        footprint = brick.footprint.ToNurbsCurve()
        cnt = rg.AreaMassProperties.Compute(footprint).Centroid
        cur_brick_loc = cnt
        trans = create_random_loc(cur_brick_loc)
        brick.move(trans)

def create_random_loc(originPt):
    randomX = round(random.random(), 5)
    randomY = round(random.random(), 5)
    newVec = rg.Vector3d(randomX*0.5, randomY*0.5, 0)
    trans = rg.Transform.Translation(newVec)
    return trans

def get_all_intersections(bricks):
    intersections = []
    for brick in bricks:
        intersections.append(brick.intersection)
    return intersections

def test_bad_data(intersections):
    is_bad = False
    for intersection in intersections:
        if(intersection == True):
            is_bad = True
            return is_bad
    return is_bad


if __name__ == "__main__":
    generate(option.option)
