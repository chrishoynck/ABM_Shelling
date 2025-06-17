import geopandas as gpd
import networkx as nx

from src.schelling_model import SchellingModel
import src.visualizations as vis

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


def simulate_model(steps, model, graph):
    """
    Run the model for a fixed number of steps, capturing initial and final grid states.

    Params:
        steps (int): Number of steps to simulate.
        model (mesa.Model): Mesa model instance with a `.schedule.agents` iterable and a `step()` method.
        graph (nx.Graph): Graph of Utrecht.

    Returns:
        snapshots (list of dict): Two‐element list of state dictionaries:
            - Index 0: state at step 0
            - Index 1: state at step (steps - 1)
        Each dict maps node IDs to occupancy:
            - -1 for empty
            - 0, 1 or 2 for an agent’s type (income level)
    """
    snapshots = []
    for i in range(steps):
        # Capture grid state: -1 empty, 0 or 1 agent types
        state = {node: -1 for node in graph.nodes}
        for agent in model.schedule.agents:
            state[agent.pos] = agent.type
        if i == 0:
            snapshots.append(state)
        elif i == steps-1:
            snapshots.append(state)
        model.step()
    return snapshots



# Simulation parameters
density = 0.8
population_dist = [0.3, 0.6, 0.1] # Low - Medium - High income people
homophily = 0.5
steps = 200
num_runs = 1

# Read geopandas file
grid_cells = gpd.read_file('city_files/utrecht.shp')
grid_cells = grid_cells.set_index('cell_id', drop=False)
grid_cells['neighbors'] = None

# Initialize and run model
graph = create_graph_Utrecht(grid_cells)
model = SchellingModel(graph, density, population_dist, homophily)
snapshots = simulate_model(steps, model, graph)
vis.visualize_start_end(population_dist, grid_cells, snapshots)


# Happyness metric, kan lang duren

# happy_agents = np.zeros((num_runs, steps))
# for j in range(num_runs):
#     model = SchellingModel(graph, density, population_dist, homophily)
#     for i in range(steps):
#         happy_agents[j, i] = sum(1 for a in model.schedule.agents if a.happy == True) / len(model.schedule.agents)
#         model.step()

# mean_happy = np.mean(happy_agents, axis=0)
# CI_happy = 1.96 * np.std(happy_agents, axis=0) / np.sqrt(num_runs)

# plt.figure(figsize=(7,5))
# plt.title('Fraction of Happy People vs. Time', fontsize=17)
# plt.plot(range(steps), mean_happy, color='blue')
# plt.fill_between(range(steps), mean_happy - CI_happy, mean_happy + CI_happy, alpha=0.3, color='blue')
# plt.xlabel('Time Steps', fontsize=15)
# plt.ylabel('Fraction of Happy People', fontsize=15)
# plt.grid(ls='dashed')
# plt.show()