from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from src.schelling_agent import SchellingAgent


class Schelling(Model):
    """Model class for the Schelling segregation model."""

    def __init__(
        self,
        height: int = 20,
        width: int = 20,
        density: float = 0.8,
        minority_pc: float = 0.5,
        homophily: float = 0.4,
        radius: int = 1,
        seed=None,
    ):
        """Create a new Schelling model.

        Args:
            width: Width of the grid
            height: Height of the grid
            density: Initial chance for a cell to be populated (0-1)
            minority_pc: Chance for an agent to be in minority class (0-1)
            homophily: Minimum number of similar neighbors needed for happiness
            radius: Search radius for checking neighbor similarity
            seed: Seed for reproducibility
        """
        super().__init__(seed=seed)

        # Model parameters
        self.density = density
        self.minority_pc = minority_pc
        self.schedule = RandomActivation(self)
        # Initialize grid
        self.grid = MultiGrid(width, height, torus=True)

        # Track happiness
        self.happy = 0

        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={
                "happy": "happy",
                "pct_happy": lambda m: (m.happy / m.schedule.get_agent_count()) * 100
                if m.schedule.get_agent_count() > 0
                else 0,
                "population": lambda m: m.schedule.get_agent_count(),
                "minority_pct": lambda m: (
                    sum(1 for a in m.schedule.agents if a.type == 1)
                    / m.schedule.get_agent_count()
                    * 100
                    if m.schedule.get_agent_count()> 0
                    else 0
                ),
            },
            agent_reporters={"agent_type": "type"},
        )

        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.random.random() < self.density:
                    # pick type
                    agent_type = 1 if self.random.random() < self.minority_pc else 0
                    # create with a proper unique_id
                    a = SchellingAgent(self.next_id(), self, agent_type,
                                       homophily=homophily,
                                       radius=radius)
                    # add to scheduler
                    self.schedule.add(a)
                    # place on the grid
                    self.grid.place_agent(a, (x, y))

        self.collect_happyness()
        self.datacollector.collect(self)

    def step(self):
        """Run one step of the model."""

        self.schedule.step()
        self.collect_happyness()
        self.datacollector.collect(self)  # Collect data
        self.running = (self.happy < self.schedule.get_agent_count())  # Continue until everyone is happy
    
    def collect_happyness(self):
        self.happy = 0  # Reset counter of happy agents
        for agent in self.schedule.agents:
            agent.assign_state()
        



# from mesa.time import RandomActivation
# from mesa import Model
# from src.schelling_agent import SchellingAgent
# import numpy as np
# from mesa.space import NetworkGrid
# from mesa.datacollection import DataCollector

# class SchellingModel(Model):
#     """Model class for the Schelling segregation model."""
#     def __init__(
#         self,
#         graph,
#         density,
#         minority_1,
#         minority_2,
#         homophily,
#         seed=None,
#     ):
#         """
#         Schelling segregation model on a network graph.

#         Params:
#             graph (nx.Graph): NetworkX graph defining adjacency of cells.
#             density (float): Probability of a node being initially occupied.
#             population_dist (list of float): Proportions of each agent type, summing to 1.
#             homophily (float): Fraction of similar neighbours required for an agent to be happy.
#             seed (int or None): Random seed for reproducibility.
#         """
#         super().__init__(seed)
#         self.schedule = RandomActivation(self)
#         self.G = graph
#         self.G._model = self
#         major = 1.0 - minority_1 - minority_2
#         assert major >= 0, "minority_1 + minority_2 must be ≤ 1"
#         self.population_dist = [major, minority_1, minority_2]
#         self.happy = 0
#         self.grid = NetworkGrid(self.G)
#         # assert sum(population_dist) == 1, "Sum of population distribution should be 1"
#         uid = 0

#         self.datacollector = DataCollector(
#             model_reporters={
#                 "happy": "happy",
#                 "pct_happy": lambda m: (m.happy / m.schedule.get_agent_count()) * 100
#                 if m.schedule.get_agent_count() > 0
#                 else 0,
#                 "population": lambda m: m.schedule.get_agent_count(),
#                 "minority_1": lambda m: (
#                     sum(1 for a in m.schedule.agents if a.type == 1)
#                     / m.schedule.get_agent_count()
#                     * 100
#                     if m.schedule.get_agent_count()> 0
#                     else 0
#                 ),
#                 "minority_2": lambda m: (
#                     sum(1 for a in m.schedule.agents if a.type == 2)
#                     / m.schedule.get_agent_count()
#                     * 100
#                     if m.schedule.get_agent_count()> 0
#                     else 0
#                 ),
#             },
#             agent_reporters={"agent_type": "type"},
#         )

#         for node in self.G.nodes:
#                 if self.random.random() < density:
#                     uid = self.next_id()
#                     agent_type = np.random.choice([0, 1, 2], p=self.population_dist)
#                     if agent_type == 0:
#                         agent = SchellingAgent(uid, self, agent_type, homophily)
#                     elif agent_type == 1:
#                         agent = SchellingAgent(uid, self, agent_type, homophily)
#                     else:
#                         agent = SchellingAgent(uid, self, agent_type, homophily)
#                     agent = SchellingAgent(uid, self, agent_type, homophily)
#                     self.grid.place_agent(agent, node)
#                     self.schedule.add(agent)
#                     # uid += 1
#         self.datacollector.collect(self)

#     def step(self):
#         """
#         Advance the model by one time step.
#         Activates all agents once in random order.
#         """
#         self.schedule.step()
#         self.happy = sum(1 for a in self.schedule.agents if a.happy)
#         self.datacollector.collect(self)  # Collect data
#         self.running = (self.happy < self.schedule.get_agent_count()) 