import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.space import NetworkGrid
from mesa.time import RandomActivation
import geopandas as gpd
import networkx as nx
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatch

class SchellingAgent(Agent):
    """Simple Schelling segregation agent."""
    def __init__(self, unique_id, model, agent_type, homophily=0.4):
        super().__init__(unique_id, model)
        self.type = agent_type
        self.homophily = homophily
        self.happy = False

    def step(self):
        neighbor_nodes = self.model.grid.get_neighbors(self.pos)
        neighbor_agents = []
        for node_id in neighbor_nodes:
            neighbor_agents.extend(
                self.model.grid.get_cell_list_contents([node_id])
            )
        if neighbor_agents:
            similar = sum(1 for n in neighbor_agents if n.type == self.type)
            frac = similar / len(neighbor_agents)
        else:
            frac = 0
        self.happy = frac >= self.homophily
        if not self.happy:
            empties = [id for id in self.model.grid.G.nodes if self.model.grid.is_cell_empty(id)]
            if empties:
                new_node = self.random.choice(empties)
                self.model.grid.move_agent(self, new_node)

class SchellingModel(Model):
    """Model class for the Schelling segregation model."""
    def __init__(self, graph, density, population_dist, homophily, seed=None):
        super().__init__(seed)
        self.schedule = RandomActivation(self)
        self.G = graph
        self.grid = NetworkGrid(self.G)
        assert sum(population_dist) == 1, "Sum of population distribution should be 1"
        uid = 0
        for node in self.G.nodes:
                if self.random.random() < density:
                    agent_type = np.random.choice([0, 1, 2], p=population_dist)
                    if agent_type == 0:
                        agent = SchellingAgent(uid, self, agent_type, homophily)
                    elif agent_type == 1:
                        agent = SchellingAgent(uid, self, agent_type, homophily)
                    else:
                        agent = SchellingAgent(uid, self, agent_type, homophily)
                    agent = SchellingAgent(uid, self, agent_type, homophily)
                    self.grid.place_agent(agent, node)
                    self.schedule.add(agent)
                    uid += 1

    def step(self):
        self.schedule.step()

# Read geopandas file
grid_cells = gpd.read_file('city_files/utrecht.shp')
grid_cells = grid_cells.set_index('cell_id', drop=False)
grid_cells['neighbors'] = None

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

# Simulation parameters
density = 0.8
population_dist = [0.3, 0.6, 0.1] # Low - Medium - High income people
homophily = 0.5
steps = 200
num_runs = 1

# Initialize and run model
model = SchellingModel(graph, density, population_dist, homophily)
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

cmap = ListedColormap(["lightgrey", "steelblue", "indianred", "yellowgreen"])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for ax, snap, title in ((ax1, snapshots[0], "Initial State"),(ax2, snapshots[-1], "Final State"),):
    # Each cell in the Utrecht grid will get its agent value, which in turn gets its own color
    grid_cells["value"] = grid_cells.index.map(lambda cid: snap.get(cid, -1))
    grid_cells.plot(
        column="value",
        categorical=True,
        cmap=cmap,
        linewidth=0.2,
        edgecolor="white",
        ax=ax,
        legend=False
    )
    ax.set_title(title)
    ax.set_axis_off()

# Custom legend
categories = [
    ("Empty", "lightgrey"),
    (f"Low Income ({population_dist[0]*100:.1f}%)", "steelblue"),
    (f"Medium Income ({population_dist[1]*100:.1f}%)", "indianred"),
    (f"High Income ({population_dist[2]*100:.1f}%)", "yellowgreen"),
]
handles = [mpatch.Patch(color=c, label=l) for l, c in categories]
ax1.legend(
    handles=handles,
    title="Category",
    loc="lower right",
    frameon=True,
    fontsize="small"
)

plt.tight_layout()
plt.show()


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