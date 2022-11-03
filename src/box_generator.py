import rhinoinside
rhinoinside.load()
import System
import Rhino
import Rhino.Geometry as rg
import rule
import generator
import box
import math
import create_edge_json
import create_node_json
import option

# loop 5000 data points
for i in range(0,option.option):

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
    brick_red = {"depth":6, "width":8, "type":"A1"}
    brick_green = {"depth":15, "width":8, "type":"B1"}
    brick_blue = {"depth":6, "width":16, "type":"C1"}
    brick_data = [brick_red, brick_green, brick_blue]
    target_n = math.floor(option / 1000) + 4
    start_plane = rg.Plane.WorldXY
    stack_bricks = cur_generator(brick_data, rules, target_n, start_plane, option, False, rule_seq)
    stack_bricks.generate()

    # All the stuff below are for visualization
    planes = [brick.origin for brick in stack_bricks.bricks]
    footprints = [brick.footprint for brick in stack_bricks.bricks]
    used_ports = stack_bricks.connected_ports
    used_bricks = stack_bricks.connected_bricks
    connected_ports_name = stack_bricks.connected_ports_name
    grow_rule_seq = stack_bricks.grow_rule_seq

    bricks = stack_bricks.bricks

    edge_json = create_edge_json.Edge_json(bricks, 12, 12, rg.Point3d(0,0,0), option)
    edge_json.parse_path()
    edge_json.save_json_data()

    node_json = create_node_json.Node_json(bricks, 12, 12, rg.Point3d(0,0,0), option)
    node_json.parse_path()
    node_json.save_json_data()

    print("currently creating data point {}".format(i))