# from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule, NetworkModule
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.ModularVisualization import ModularServer
from src.schelling_model import SchellingModel
import networkx as nx
import geopandas as gpd

def create_graph_Utrecht(grid_cells):
    """
    Create a graph of adjacent cells from a GeoDataFrame of Utrecht grid cells.
    Edges connect any two cells whose polygons touch.

    Params:
        grid_cells (GeoDataFrame): holds the topology of the grid

    Returns:
        graph (nx.Graph): An undirected graph where each node is a cell identifier and
        each edge represents two neighbouring cells.
    """
    # Create network of the Utrecht grid
    sindex = grid_cells.sindex
    graph = nx.Graph()
    for cell_id, cell in grid_cells.iterrows():
        # find all polygons whose bounding‐boxes intersect
        possible_matches_index = list(sindex.intersection(cell.geometry.bounds))
        possible = grid_cells.iloc[possible_matches_index]
        # refine to those which actually touch
        real_neighbors = possible[possible.touches(cell.geometry)]
        for nbr_id in real_neighbors.cell_id:
            graph.add_edge(cell_id, nbr_id)
    return graph


# 1) Define a network-portrayal function
def network_portrayal(G):
    model = G._model
    W, H    = 500, 500   # must match your NetworkModule canvas_width/height
    colour_map = {0: "blue", 1: "orange", 2: "red"}
    nodes, links = [], []

    for node in G.nodes():
        # lookup static position
        ux, uy = G.nodes[node].get('coords', (0.5, 0.5))
        x_pix = ux * W
        y_pix = (1 - uy) * H   # invert Y so north is “up”

        agents = model.grid.get_cell_list_contents([node])
        if agents:
            a     = agents[0]
            colour= colour_map[a.type]
            size  = 6
        else:
            colour= "lightgrey"
            size  = 4

        nodes.append({
            "id":    node,
            "x":     x_pix,
            "y":     y_pix,
            "color": colour,
            "size":  size,
            "label": ""
        })

    for u, v in G.edges():
        links.append({"source": u, "target": v})

    return {"nodes": nodes, "links": links}

# 1) Read & build your graph
grid_cells = gpd.read_file('city_files/utrecht.shp')
grid_cells = grid_cells.set_index('cell_id', drop=False)
grid_cells['neighbors'] = None

G = create_graph_Utrecht(grid_cells)

minx, miny, maxx, maxy = grid_cells.total_bounds

# attach a 'coords' attr in [0,1]×[0,1] space
pos = {}
for cell_id, row in grid_cells.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    # normalise to 0–1
    nx_ = (cx - minx) / (maxx - minx)
    ny_ = (cy - miny) / (maxy - miny)
    pos[cell_id] = (nx_, ny_)
nx.set_node_attributes(G, pos, 'coords')


# 2) User sliders, including graph
model_params = {
    "graph": G,
    "density": UserSettableParameter("slider", "Initial Density", 0.8, 0.1, 1.0, 0.05),
    "minority_1": UserSettableParameter("slider", "Mid-income %", 0.2, 0.0, 1.0, 0.05),
    "minority_2": UserSettableParameter("slider", "Low-income %", 0.5, 0.0, 1.0, 0.05),
    "homophily":  UserSettableParameter("slider", "Homophily Threshold", 0.4, 0.0, 1.0, 0.05),
}


# 2) Instantiate the NetworkModule
network = NetworkModule(
    portrayal_method=network_portrayal,
    canvas_width=500,    # pick whatever size you like
    canvas_height=500,
    library="d3"         # use d3 for force-directed layout
)

chart = ChartModule([
    {"Label": "pct_happy", "Color": "Green"}
])


# 3) Plug into your server in place of CanvasGrid
server = ModularServer(
    SchellingModel,
    [network, chart],
    "Schelling Segregation Model",
    model_params
)
server.port = 8521
server.launch()