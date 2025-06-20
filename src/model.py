import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from src.agent import SchellingAgent
from mesa.datacollection import DataCollector

class District:
    def __init__(self, id_, rent=0):
        """
        Initialize a District.

        Params:
            id_ (int): Unique district identifier.
        """
        self.id = id_
        self.counts = {0: 0, 1: 0, 2: 0}
        self.action_counts = {'a': 0, 'b': 0, 'c': 0}
        self.empty_places = []
        self.bids = []
        self.area = 0
        self.rent = rent

    def count_of(self, group):
        """
        Get the number of agents of a given type in this district.

        Params:
            group (int): Agent type index.

        Returns:
            int: Count of agents of that type.
        """
        return self.counts[group]

    def count_of_actions(self, action):
        """
        Get the number of agents currently taking a given action in this district.

        Params:
            action (str): Action label ('a', 'b', or 'c').

        Returns:
            int: Count of agents performing that action.
        """
        return self.action_counts[action]

    def total_in_dist(self):
        """
        Compute the total number of agents in this district.

        Returns:
            int: Sum of all agent counts across types.
        """
        waarde = 0
        for i in range(3):
            waarde += self.count_of(i)
        return waarde

    def change_actions(self, old_action, new_action):
        """
        Update action counts when an agent switches actions.

        Params:
            old_action (str): Previous action label.
            new_action (str): New action label.
        """
        self.action_counts[old_action] -= 1
        self.action_counts[new_action] += 1

    def move_in(self, agent):
        """
        Register an agent moving into this district.

        Params:
            agent (Agent): The agent entering; must have 'pos', 'type' and 'current_action'.
        """
        self.empty_places.remove(agent.pos)
        self.action_counts[agent.current_action] += 1
        self.counts[agent.type] += 1

    def move_out(self, agent):
        """
        Register an agent moving out of this district.

        Params:
            agent (Agent): The agent leaving; must have 'pos', 'type' and 'current_action'.
        """
        self.empty_places.append(agent.pos)
        self.action_counts[agent.current_action] -= 1
        self.counts[agent.type] -= 1

class SchellingModel(Model):
    """Model class for the Schelling segregation model."""
    def __init__(self, width, height, density, p_random, pay_c, pay_m, max_tenure, u_threshold, alpha, population_distribution, seed=None, num_districts = 3):
        """Create a new Schelling model.

        Args:
            width: Width of the grid
            height: Height of the grid
            density: Initial chance for a cell to be populated (0-1)
            population_distribution: Chance for an agent to be in a class (0-1-2)
            homophilies: Minimum number of similar neighbors needed for happiness
            seed: Seed for reproducibility
        """
        super().__init__(seed)
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, torus=True)

        # measuring happyness per type 
        self.happy = 0
        self.alpha = alpha
        self.happiness_per_type = [0.0, 0.0, 0.0]
        self.num_agents_per_type = [0, 0, 0]

        self.pay_c = pay_c
        self.pay_m = pay_m
        self.mu = 0
        self.max_tenure = max_tenure
        self.u_threshold = u_threshold
        self.p_random = p_random

        # set districts 
        stripe_width = width // num_districts
        self.districts = [
            District(i) for i in range(num_districts)
                ]
        self.district_of = {
            (x, y): self.districts[min(x // stripe_width, num_districts - 1)]
            for x in range(width) for y in range(height)
        }
        uid = 0
        assert sum(population_distribution) == 1, "Sum of income distribution should be 1"
        
        district_areas = self.outline_districts(width, height)
        # initialize agents on grid 
        for x in range(width):
            for y in range(height):

                for district_id, boxes in district_areas.items():
                    if any(self.point_in_box(x, y, box) for box in boxes):
                        district = self.districts[district_id]
                        self.district_of[(x, y)] = district
                        district.empty_places.append((x, y))
                        district.area +=1
                        break
                else:
                    raise ValueError(f"cell ({x,y}) does not fall in any district ")
                
                if self.random.random() < density:
                    agent_type = np.random.choice([0, 1, 2], p=population_distribution)
                    if agent_type == 0:
                        agent = SchellingAgent(uid, self, agent_type)
                    elif agent_type == 1:
                        agent = SchellingAgent(uid, self, agent_type)
                    else:
                        agent = SchellingAgent(uid, self, agent_type)
                
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)

                    # add agent to district 
                    district.move_in(agent)
                    uid += 1
                    self.num_agents_per_type[agent_type] += 1
        
        
        self.calc_mu()
        self.set_inital_rents(self.mu)

        #  metrics
        self.metrics = (self.exposure_to_others(self.num_agents_per_type, self.districts), 
                        self.dissimilarity(self.num_agents_per_type, self.districts)) 
        
        assert sum(self.num_agents_per_type) == self.schedule.get_agent_count(), "total number of agents across types is not the same as total agents"
        
        # collect initial data
        # self.datacollector.collect(self)

    def calc_mu(self):
        mu = dict()
        for d in self.districts:
            if d.total_in_dist():
                mu[d] = {
                    'a': d.action_counts['a'] / d.total_in_dist(),
                    'b': d.action_counts['b'] / d.total_in_dist(),
                    'c': d.action_counts['c'] / d.total_in_dist()
                }
            else:
                mu[d] = {
                    'a': 0,
                    'b': 0,
                    'c': 0
                }
        self.mu= mu

    def set_inital_rents(self, mu):

        # get bids on districts 
        for ag in self.schedule.agents:
            _ = ag.get_best_district(mu)

    def point_in_box(self, x, y, box):
            x1, y1, x2, y2 = box
            return x1 <= x < x2 and y1 <= y < y2

    def outline_districts(self, width, height):
        district_areas = {
            0: [
                (0, 0, int(width * 0.4), int(height * 0.2)),
                (0, int(height * 0.2), int(width * 0.5), height),
            ],
            1: [
                (int(width * 0.5), int(height * 0.2), width , height ),
                (int(width * 0.4), 0, int(width * 0.6), int(height * 0.2)),
                (int(width * 0.5), int(height * 0.2), int(width * 0.8), int(height * 0.3)),
                (int(width * 0.6), int(height * 0.1), int(width * 0.7), int(height * 0.2)),
            ],
            2: [
                (int(width * 0.7), 0, width, int(height * 0.2)),
                (int(width * 0.6), 0, int(width * 0.7), int(height * 0.1)),
                (int(width * 0.8), int(height * 0.2), width , int(height * 0.3)),
            ],
        }
        return district_areas

        
    
    def dissimilarity(self, typetjes, districtjes): 
        """
        Compute the pairwise-average dissimilarity index.

        Args:
            typetjes (list[int]): Total count per agent type.
            districtjes (list): Districts with a `count_of(type)` method.

        Returns:
            float: Dissimilarity index in [0, 1].
        """
        diss_per_type = []

        # avoid types with that no person has 
        type_pairs = [
        (i, j)
        for i, j in [(0,1), (0,2), (1,2)]
        if typetjes[i] > 0 and typetjes[j] > 0
        ]
        # return if no agents in the grid 
        if len(type_pairs) == 0: 
            return 0

        # loop through pairs to accumulate dissimilarity
        for type1, type2 in  type_pairs:
            waarde = 0

            # loop through districts 
            for d in districtjes: 
                waarde += np.abs(d.count_of(type1)/typetjes[type1]-
                                 d.count_of(type2)/typetjes[type2])
            diss_per_type.append(0.5*waarde)
            
        return np.mean(diss_per_type)
            

    def exposure_to_others(self, typetjes, districtjes):
        """
        Compute the overall population-weighted exposure to other types.

        Params:
            typetjes (list[int]): Total count per agent type.
            districtjes (list): Districts with a `count_of(type)` method.

        Returns:
            float: Exposure index in [0, 1].
        """

        exposure = 0
        total_agents = self.schedule.get_agent_count()

        # iterate over all types
        for i in range(len(typetjes)):
            tot_i = typetjes[i]

            # if type count is 0, skip this type 
            if  tot_i == 0:
                continue 

            exposed_i = 0.0
            for d in districtjes:
                ni    = d.count_of(i)
                total_in_district  = sum(d.count_of(j) for j in range(len(typetjes)))

                # pass if number in district is 0 
                if total_in_district == 0:
                    continue

                # exposure of type i in district d 
                exposed_i += (ni / tot_i) * ((total_in_district - ni) / total_in_district)

            # add normalized value
            # exposure += (tot_i/total_agents)*exposed_i
            exposure += 1/3*exposed_i

        return exposure
    
    def set_rent_districts(self):
        total_people = 0
        for d in self.districts:
            d.bids.sort(key=lambda x: x[1], reverse=True)
            supply = d.area
            
            # print(len(d.bids))
            if len(d.bids) >= supply:
                rent = d.bids[supply - 1][1]
            else:
                rent = d.bids[-1][1] if d.bids else 0
                if not d.bids:
                    raise ValueError("there are no biddings on this district, should not happen")
            d.rent = rent

            # if d.id == 0: 
            #     d.rent = 2
            # elif d.id == 1: 
            #     d.rent = 5
            # elif d.id == 2: 
            #     d.rent = 10
            # else:
            #     raise ValueError("id's not correctly set")

            # print(f"{d.id}: rent {d.rent}")

            d.bids = []
            total_people += d.total_in_dist()
        
         # make sure the dynamical update results in same measures over time 
        assert total_people == self.schedule.get_agent_count(), f"Number of agents over districts ({total_people}) does not equal total number ({self.schedule.get_agent_count()})"

    


    def step(self):

        # reset happiness
        
        self.happy = 0
        self.happiness_per_type = [0.0, 0.0, 0.0]
        
        self.metrics = (self.exposure_to_others(self.num_agents_per_type, self.districts), 
                        self.dissimilarity(self.num_agents_per_type, self.districts)) 

        self.calc_mu()
        self.set_rent_districts()
       
        #  agent step
        self.schedule.step()
        
        # print(self.happiness_per_type)
        # collect data
        # self.datacollector.collect(self) 


# Set up data collection
        # self.datacollector = DataCollector(
        #     model_reporters={
        #         "happy": "happy",
        #         "pct_happy": lambda m: (m.happy / m.schedule.get_agent_count()) * 100
        #         if m.schedule.get_agent_count() > 0
        #         else 0,
        #         "population": lambda m: m.schedule.get_agent_count(),
        #         "minority_pct_1": lambda m: (
        #             sum(1 for a in m.schedule.agents if a.type == 1)
        #             / m.schedule.get_agent_count()
        #             * 100
        #             if m.schedule.get_agent_count()> 0
        #             else 0
        #         ),
        #         "minority_pct_2": lambda m: (
        #             sum(1 for a in m.schedule.agents if a.type == 2)
        #             / m.schedule.get_agent_count()
        #             * 100
        #             if m.schedule.get_agent_count()> 0
        #             else 0
        #         ),
        #     },
        #     agent_reporters={"agent_type": "type"},
        # )