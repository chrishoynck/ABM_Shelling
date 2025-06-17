import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from src.agent import SchellingAgent

class SchellingModel(Model):
    """Model class for the Schelling segregation model."""
    def __init__(self, width, height, density, income_distribution, homophilies, radius, seed=None):
        super().__init__(seed)
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, torus=True)
        uid = 0
        assert sum(income_distribution) == 1, "Sum of income distribution should be 1"
        homophily_low, homophily_medium, homophily_high = homophilies
        for x in range(width):
            for y in range(height):
                if self.random.random() < density:
                    agent_type = np.random.choice([0, 1, 2], p=income_distribution)
                    if agent_type == 0:
                        agent = SchellingAgent(uid, self, agent_type, homophily_low, radius)
                    elif agent_type == 1:
                        agent = SchellingAgent(uid, self, agent_type, homophily_medium, radius)
                    else:
                        agent = SchellingAgent(uid, self, agent_type, homophily_high, radius)
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)
                    uid += 1

    def step(self):
        self.schedule.step()