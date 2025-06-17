import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from src.agent import SchellingAgent
from mesa.datacollection import DataCollector

class SchellingModel(Model):
    """Model class for the Schelling segregation model."""
    def __init__(self, width, height, density, income_distribution, homophilies, radius, seed=None):
        """Create a new Schelling model.

        Args:
            width: Width of the grid
            height: Height of the grid
            density: Initial chance for a cell to be populated (0-1)
            income_distribution: Chance for an agent to be in a class (0-1-2)
            homophilies: Minimum number of similar neighbors needed for happiness
            radius: Search radius for checking neighbor similarity
            seed: Seed for reproducibility
        """
        super().__init__(seed)
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, torus=True)

        # measuring happyness per type 
        self.happy = 0
        self.happiness_per_type = [0.0, 0.0, 0.0]
        self.num_agents_per_type = [0, 0, 0]

        uid = 0
        assert sum(income_distribution) == 1, "Sum of income distribution should be 1"
        homophily_low, homophily_medium, homophily_high = homophilies

        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={
                "happy": "happy",
                "pct_happy": lambda m: (m.happy / m.schedule.get_agent_count()) * 100
                if m.schedule.get_agent_count() > 0
                else 0,
                "population": lambda m: m.schedule.get_agent_count(),
                "minority_pct_1": lambda m: (
                    sum(1 for a in m.schedule.agents if a.type == 1)
                    / m.schedule.get_agent_count()
                    * 100
                    if m.schedule.get_agent_count()> 0
                    else 0
                ),
                "minority_pct_2": lambda m: (
                    sum(1 for a in m.schedule.agents if a.type == 2)
                    / m.schedule.get_agent_count()
                    * 100
                    if m.schedule.get_agent_count()> 0
                    else 0
                ),
            },
            agent_reporters={"agent_type": "type"},
        )

        # initialize agents on grid 
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

                    self.num_agents_per_type[agent_type] += 1

        # collect initial data
        self.datacollector.collect(self)

    def step(self):
        # reset happiness
        self.happy = 0
        self.happiness_per_type = [0.0, 0.0, 0.0]

        #  agent step
        self.schedule.step()

        # collect data
        self.datacollector.collect(self) 


