import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from src.agent import SchellingAgent
from mesa.datacollection import DataCollector

class District:
    def __init__(self, id_):
        self.id = id_
        self.counts = {0:0, 1:0, 2:0}
        self.total = [0, 0, 0]
    def count_of(self, group):
        return self.counts[group]
    
class SchellingModel(Model):
    """Model class for the Schelling segregation model."""
    def __init__(self, width, height, density, income_distribution, homophilies, radius, seed=None, num_districts = 3):
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

        # set districts 
        stripe_width = width // num_districts
        self.districts = [
            District(i) for i in range(num_districts)
                ]
        self.district_of = {
            (x, y): self.districts[min(x // stripe_width, num_districts - 1)]
            for x in range(width) for y in range(height)
        }

        #  metrics
        self.metrics = (self.exposure_to_others(), self.dissimilarity()) 
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
                    
                    self.district_of[(x,y)].counts[agent_type] +=1
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)
                    uid += 1

                    self.num_agents_per_type[agent_type] += 1
        
        for d in self.districts: 
            d.total = self.num_agents_per_type

        # collect initial data
        self.datacollector.collect(self)
    
    def dissimilarity(self): 
        diss_per_type = []

        # avoid types with that no person has 
        type_pairs = [
        (i, j)
        for i, j in [(0,1), (0,2), (1,2)]
        if self.num_agents_per_type[i] > 0 and self.num_agents_per_type[j] > 0
        ]

        # return if no agents in the grid 
        if len(type_pairs) == 0: 
            return 0

        # loop through pairs to accumulate dissimilarity
        for type1, type2 in  type_pairs:
            waarde = 0

            # loop through districts 
            for d in self.districts: 
                waarde += np.abs(d.count_of(type1)/self.num_agents_per_type[type1]-
                                 d.count_of(type2)/self.num_agents_per_type[type2])
            diss_per_type.append(0.5*waarde)
            
        return np.mean(diss_per_type)
            

    def exposure_to_others(self):

        exposure = 0

        total_agents = self.schedule.get_agent_count()

        for i in range(len(self.num_agents_per_type)):
            tot_i = self.num_agents_per_type[i]
            if  tot_i == 0:
                continue 

            exposed_i = 0.0
            for d in self.districts:
                ni    = d.count_of(i)
                total_in_district  = sum(d.count_of(j) for j in range(len(self.num_agents_per_type)))
                if total_in_district == 0:
                    continue
                exposed_i += (ni / tot_i) * ((total_in_district - ni) / total_in_district)
            exposure += (tot_i/total_agents)*exposed_i
        return exposure

    def step(self):
        # reset happiness
        self.happy = 0
        self.happiness_per_type = [0.0, 0.0, 0.0]

        #  agent step
        self.schedule.step()

        # collect data
        self.datacollector.collect(self) 


