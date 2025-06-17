from mesa.time import RandomActivation
from mesa import Model
from src.schelling_agent import SchellingAgent
import numpy as np
from mesa.space import NetworkGrid

class SchellingModel(Model):
    """Model class for the Schelling segregation model."""
    def __init__(self, graph, density, population_dist, homophily, seed=None):
        """
        Schelling segregation model on a network graph.

        Params:
            graph (nx.Graph): NetworkX graph defining adjacency of cells.
            density (float): Probability of a node being initially occupied.
            population_dist (list of float): Proportions of each agent type, summing to 1.
            homophily (float): Fraction of similar neighbours required for an agent to be happy.
            seed (int or None): Random seed for reproducibility.
        """
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
        """
        Advance the model by one time step.
        Activates all agents once in random order.
        """
        self.schedule.step()