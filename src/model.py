import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from src.agent import SchellingAgent
from src.district import District
import random
# from mesa.datacollection import DataCollector


# class SeededActivation(RandomActivation):
#     def step(self):
#         # shuffle with the *model’s* random.Random
#         self.model.random.shuffle(self.agents)
#         for a in list(self.agents):
#             a.step()

class SchellingModel(Model):
    """Model class for the Schelling segregation model."""
    def __init__(self, width, height, density, p_random, pay_c, pay_m, max_tenure, u_threshold, alpha, population_distribution, income_dist, seedje=None, num_districts = 3):
        """
        Initialise a new Schelling model with spatial districts and bidding.

        Params:
            width (int): Width of the grid (number of columns).
            height (int): Height of the grid (number of rows).
            density (float): Probability in [0,1] that a given cell starts occupied.
            p_random (float): Probability of a random move when tenure expires.
            pay_c (float): Payoff for coordination in the neighbourhood game.
            pay_m (float): Payoff for miscoordination in the neighbourhood game.
            max_tenure (int): Number of steps an agent stays before forced reconsideration.
            u_threshold (float): Minimum average utility for an agent to remain happy.
            alpha (float): Weight on consumption (vs. expected game payoff) in bid utility.
            population_distribution (list[float]): Probabilities summing to 1 of each agent type.
            income_dist (list[float]): Base income level for each agent type.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            num_districts (int, optional): Number of spatial districts. Defaults to 3.
        """
        
        if seedje is None:
            print("Model not seeded!!")
        else: 
            seedje = int(seedje)

        random.seed(seedje)
        np.random.seed(seedje)
        # initialize grid and scheduler
        super().__init__(seedje)
        
        
    
        self.np_random = np.random.default_rng(seedje)
        self.schedule = SimultaneousActivation(self)
        self.grid = MultiGrid(width, height, torus=False)

        # measuring happyness per type 
        self.happy = 0
        self.alpha = alpha
        self.happiness_per_type = [0.0, 0.0, 0.0]
        self.num_agents_per_type = [0, 0, 0]
        
        # set model parameters
        self.pay_c = pay_c
        self.pay_m = pay_m
        self.mu = 0
        self.max_tenure = max_tenure
        self.u_threshold = u_threshold
        self.p_random = p_random

        
        self.districts = [
            District(i) for i in range(num_districts)
                ]
        self.district_of = dict()
        # using evenly distributed districts
        # set districts 
        # stripe_width = width // num_districts
        # self.district_of = {
        #     (x, y): self.districts[min(x // stripe_width, num_districts - 1)]
        #     for x in range(width) for y in range(height)
        # }
        uid = 0
        assert sum(population_distribution) == 1, "Sum of income distribution should be 1"
        
        district_areas = self.outline_districts(width, height)
        # initialize agents on grid 
        for x in range(width):
            for y in range(height):
                uid = self.set_grid(x,y, district_areas, income_dist, population_distribution, density, uid)

        #  metrics
        self.metrics = (self.exposure_to_others(self.num_agents_per_type, self.districts), 
                        self.dissimilarity(self.num_agents_per_type, self.districts)) 
        
        assert sum(self.num_agents_per_type) == self.schedule.get_agent_count(), "total number of agents across types is not the same as total agents"
        
        # collect initial data
        # self.datacollector.collect(self)
    def set_grid(self, x, y, district_areas, income_dist, population_distribution, density, uid):
        """
        Assign a cell to its district, initialise rent, and optionally place an agent.

        Params:
            x (int): X-coordinate of the grid cell.
            y (int): Y-coordinate of the grid cell.
            district_areas (dict[int, list[tuple[int,int,int,int]]]): Mapping of district IDs to their rectangle definitions.
            income_dist (list[float]): Base income for each agent type.
            population_distribution (list[float]): Probabilities of each agent type.
            density (float): Probability in [0,1] to place an agent at this cell.
            uid (int): Current unique ID counter for agents.

        Returns:
            int: Updated unique ID after possibly creating a new agent.

        Raises:
            ValueError: If (x,y) does not lie within any district’s defined boxes.
        """
        for district_id, boxes in district_areas.items():
            if any(self.point_in_box(x, y, box) for box in boxes):
                district = self.districts[district_id]
                self.district_of[(x, y)] = district

                # set rent to initial income
                if self.alpha > 0:
                    district.rent = income_dist[district_id]/3
                    district.next_rent = income_dist[district_id]/3
                    district.min = income_dist[district_id]/6
                    district.max =  income_dist[district_id]/1

                district.empty_places.append((x, y))
                district.area +=1
                break
        else:
            raise ValueError(f"cell ({x,y}) does not fall in any district ")
        
        # place agent on cell depending on density. pick agent type depending on agent distribution
        if self.np_random.random() < density:
            agent_type = self.np_random.choice([0, 1, 2], p=population_distribution)
            income = income_dist[agent_type] 
            if agent_type == 0:
                agent = SchellingAgent(uid, self, agent_type, income)
            elif agent_type == 1:
                agent = SchellingAgent(uid, self, agent_type, income)
            else:
                agent = SchellingAgent(uid, self, agent_type, income)
        
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)

            # add agent to district 
            district.move_in(agent)
            uid += 1
            self.num_agents_per_type[agent_type] += 1
        return uid

    def calc_mu(self):
        """
        Compute the fraction of agents choosing each action in every district and store the result.
        Used for determining rent and favourite district
        """
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
        return mu

    def point_in_box(self, x, y, box):
            """
            Determine whether a point lies within a given rectangular box.

            Params:
                x (int or float):  x-coordinate of the point.
                y (int or float):  y-coordinate of the point.
                box (tuple[int, int, int, int]):  (x1, y1, x2, y2) defining the rectangle.

            Returns:
                bool: True if x1 ≤ x < x2 and y1 ≤ y < y2, False otherwise.
            """
            x1, y1, x2, y2 = box
            return x1 <= x < x2 and y1 <= y < y2

    def outline_districts(self, width, height):
        """
        Compute the rectangular regions for each district given canvas dimensions.

        Params:
            width (int):   Total width in pixels.
            height (int):  Total height in pixels.

        Returns:
            dict[int, list[tuple[int, int, int, int]]]:
                Mapping each district ID to a list of (x1, y1, x2, y2) rectangles covering that district.
        """
        district_areas = {
            0: [
                (int(width * 0.7), 0, width, int(height * 0.2)),
                (int(width * 0.6), 0, int(width * 0.7), int(height * 0.1)),
                (int(width * 0.8), int(height * 0.2), width , int(height * 0.3)),
            ],
            1: [
                (int(width * 0.5), int(height * 0.2), width , height ),
                (int(width * 0.4), 0, int(width * 0.6), int(height * 0.2)),
                (int(width * 0.5), int(height * 0.2), int(width * 0.8), int(height * 0.3)),
                (int(width * 0.6), int(height * 0.1), int(width * 0.7), int(height * 0.2)),
            ],

            2: [
                (0, 0, int(width * 0.4), int(height * 0.2)),
                (0, int(height * 0.2), int(width * 0.5), height),
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
        # total_agents = self.schedule.get_agent_count()

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
        """
        Update rents for all districts based on collected bids and verify population consistency.

        Raises:
            AssertionError: if the total number of agents across districts does not equal
                            the total scheduled agents.
        """
        total_people = 0
        for d in self.districts:

            # determine rent for every district 
            d.my_rent_based_on_bids()
            total_people += d.total_in_dist()
        
         # make sure the dynamical update results in same measures over time 
        assert total_people == self.schedule.get_agent_count(), f"Number of agents over districts ({total_people}) does not equal total number ({self.schedule.get_agent_count()})"

    


    def step(self):
        """
        Advance the model by one tick:

        1. Reset overall happiness and per-type happiness counts.
        2. Recompute action fractions (mu).
        3. If alpha > 0:
            a. Clear previous bids and high_bids.
            b. Have every agent submit WTP bids for each district.
            c. Update each district’s rent via set_rent_districts().
        4. Execute each agent’s step().
        5. Recalculate exposure and dissimilarity metrics.

        """

        # reset happiness
        self.happy = 0
        self.happiness_per_type = [0.0, 0.0, 0.0]
        
        # Calculate action fractions
        self.mu = self.calc_mu()

        # Only Calculate rents if alpha is nonzero
        if self.alpha > 0:
            # 1: Remove the bids from the last round
            for d in self.districts:
                d.bids.clear()
                d.high_bids.clear()

            # 2: Every agent submits its WTP for each district
            for agent in self.schedule.agents:
                agent.bid_for_districts(self.mu, self.alpha)
                agent.choose_action()

            # 3: Set the rent of each district
            self.set_rent_districts()
        else: 
            for agent in self.schedule.agents:
                agent.choose_action()
        # Agent step
        self.schedule.step()

        self.metrics = (self.exposure_to_others(self.num_agents_per_type, self.districts), 
                        self.dissimilarity(self.num_agents_per_type, self.districts)) 
        
        ###########################################################
        # Als we rents willen gebruiken is de goede volgorde: 
        # 1) Bereken mu
        # 2) Verwijder de bids van vorige ronde
        # 3) Bereken nieuwe bids d.m.v. de willingness-to-pay
        # 4) Bepaal de nieuwe rents afhankelijk van de WTP en de supply
        # 5) Daarna maken alle agents een stap en bepalen ze voor zichzelf wat het beste district is
        ###########################################################
        # Wat we voorheen fout deden was dat we de WTP en de beste district tegelijk bepaalde,
        # dus met de volgorde was iets mis.
        # De functie get_best_district is dus nu opgedeeld in bid_for_district en choose_best_district_given_rents

        

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