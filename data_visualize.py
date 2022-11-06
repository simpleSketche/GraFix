import networkx as nx
import matplotlib.pyplot as plt
import json
import os

def read_specific_edge_json(json_path):
    with open(json_path, 'r') as f:
        return json.loads(f.read())


def create_graph(edge_json):
    G = nx.Graph()
    print(edge_json)
    adjacency = edge_json['adjacency']
    edge_index = 0
    for edge in adjacency:
        print(edge)
        G.add_edge(edge[0],edge[1], edge=edge_index)
        edge_index += 1
    return G

print(os.getcwd())
index = 4999
json_path = os.getcwd() + '\\data\\room_base_graph\\edges\\{}.json'.format(index)
print(json_path)
data = read_specific_edge_json(json_path)
G = create_graph(data)

options = {
    "font_size": 36,
    "node_size": 3000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 5,
    "width": 5,
}
nx.draw_networkx(G, **options)

# Set margins for the axes so that nodes aren't clipped
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.show()