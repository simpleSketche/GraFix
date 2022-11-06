"""
This code generates the dataset from the legacy input data that is generated from Grasshopper.

"""
import pathlib
import json
import random

from ladybug_geometry.geometry2d import Polygon2D, Point2D, Vector2D

__here__ = pathlib.Path(__file__).parent
__root__ = __here__.parent
data_folder_in = __root__.joinpath('data', 'room_base_graph')
nodes_folder = data_folder_in.joinpath('nodes_out')

data_folder_out = __root__.joinpath('data', 'vertices_as_nodes')
data_folder_out.mkdir(parents=True, exist_ok=True)

for f in nodes_folder.iterdir():
    if f.suffix != '.json':
        continue
    i = f.stem
    data = json.loads(f.read_text())
    polygons = []
    for room in data['corners']:
        room_vertices = [Point2D(corner[0], corner[1]) for corner in room]
        # remove the last vertices that is duplicated
        polygon = Polygon2D(room_vertices[:-1])
        polygons.append(polygon)

    # write the vertices for this option
    nodes_fixed_model = data_folder_out.joinpath(f'{i}_out.json')
    nodes_org_model = data_folder_out.joinpath(f'{i}_in.json')
    edges_file = data_folder_out.joinpath(f'{i}_edges.json')
    rooms_file = data_folder_out.joinpath(f'{i}_rooms.json')
    rooms_data = {}
    vertices = []
    edges = []
    vertex_count = 0
    for room_id, polygon in enumerate(polygons):
        for count, vertex in enumerate(polygon.vertices):
            vertices.append([vertex.x, vertex.y])
            if count == 0:
                rooms_data[room_id] = [vertex_count]
                vertex_count += 1
                continue
            edges.append([vertex_count - 1, vertex_count])
            rooms_data[room_id].append(vertex_count)
            vertex_count += 1

    nodes_fixed_model.write_text(json.dumps(vertices))
    edges_file.write_text(json.dumps(edges))
    rooms_file.write_text(json.dumps(rooms_data))

    # make a problematic version of the room
    vertices = []
    for room_id, polygon in enumerate(polygons):
        for vertex in polygon.vertices:
            should_change = random.randint(0, 3)
            if should_change % 2 == 0:
                # don't change anything
                x, y = vertex.x, vertex.y
            else:
                # change between 0 to 10 cm
                x_dist = random.randint(0, 10) / 100
                y_dist = random.randint(0, 10) / 100
                x_dir = random.randint(0, 1)
                y_dir = random.randint(0, 1)
                vector = Vector2D(
                    x = x_dist if x_dir == 0 else -x_dist,
                    y = y_dist if y_dir == 0 else -y_dist
                )
                vertex = vertex.move(vector)
                x, y = vertex.x, vertex.y

            vertices.append([x, y])
    
    nodes_org_model.write_text(json.dumps(vertices))