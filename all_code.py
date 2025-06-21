from mesa import Agent
import numpy as np

class SchellingAgent(Agent):
    """Simple Schelling segregation agent."""
    def __init__(self, unique_id, model, agent_type):
        super().__init__(unique_id, model)
        self.type = agent_type

        # initial utilities of playing an action 
        self.U_A = self.model.pay_c if agent_type==0 else self.model.pay_m # A = 1
        self.U_B = self.model.pay_c if agent_type==1 else self.model.pay_m # B = 2
        self.U_C = self.model.pay_c if agent_type==2 else self.model.pay_m # C = 3

        # initial number of having an action played 
        self.N_A = self.N_B = self.N_C = 10

        base_income = {0: 6, 1: 10, 2: 14}[agent_type]
        self.income = base_income * np.random.lognormal(0, 0.25)

        self.current_action = np.random.choice(['a', 'b', 'c'])
        self.tenure = 0
        self.utility_sum = 0
        self.happy = False
    
    def choose_action(self):
        """
        Sample a new action from the softmax of (U_A, U_B, U_C) and update district records.
        """

        utils = np.array([self.U_A, self.U_B, self.U_C])
        exps  = np.exp(utils)
        probs = exps / exps.sum()

        # dynamically update the actions taken within district 
        my_district = self.model.district_of[self.pos]
        old_action = self.current_action
        self.current_action = np.random.choice(['a','b','c'], p=probs)
        my_district.change_actions(old_action, self.current_action)

    def update_utility_after_games(self, new_payoff):
        """
        Play the coordination game with all neighbors and return the average payoff.

        Returns:
            float: Mean payoff across all neighboring interactions.
        """
        if self.current_action == 'a':
            self.U_A = (self.N_A * self.U_A + new_payoff) / (self.N_A + 1)
            self.N_A += 1
        elif self.current_action == 'b':
            self.U_B = (self.N_B * self.U_B + new_payoff) / (self.N_B + 1)
            self.N_B += 1
        else:
            self.U_C = (self.N_C * self.U_C + new_payoff) / (self.N_C + 1)
            self.N_C += 1
    
    
    def average_utility(self):
        """Average utility for an agent over the time that they have spent in the same place."""
        return self.utility_sum / self.tenure if self.tenure > 0 else 0

    def play_game(self):
        """
        Compute the mean payoff of playing a coordination game with neighbours.

        Returns:
            float: Average payoff over all neighbouring interactions.
        """

        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=1
        )
        # Play the coordination game with each neighbour
        total_payoff_games = 0
        mean_payoff = 0
        if neighbors:
            for n in neighbors:
                if self.current_action == n.current_action:
                    payoff = self.model.pay_c
                else:
                    payoff = self.model.pay_m
                total_payoff_games += payoff
            mean_payoff = total_payoff_games / len(neighbors) if len(neighbors) != 0 else 0
        return mean_payoff

    def bid_for_districts(self, mu): 
        """
        Phase 1: Compute and store each agents bid (WTP) for each district. Does not move the agent or choose the best district yet.
        """
        expA, expB, expC = np.exp(self.U_A), np.exp(self.U_B), np.exp(self.U_C)
        total = expA + expB + expC
        Probs = {'a': expA/total, 'b': expB/total, 'c':expC/total}
        for district in self.model.districts:
            E_payoff = (Probs['a'] * (self.model.pay_c*mu[district]['a'] + self.model.pay_m*(1-mu[district]['a'])) + 
                        Probs['b'] * (self.model.pay_c*mu[district]['b'] + self.model.pay_m*(1-mu[district]['b'])) + 
                        Probs['c'] * (self.model.pay_c*mu[district]['c'] + self.model.pay_m*(1-mu[district]['c'])))
            
            rent = district.rent
            consumption = self.income - min(rent, self.income)

            WTP = (consumption**self.model.alpha) * (E_payoff**(1-self.model.alpha))

            district.bids.append((self, WTP))


    def choose_best_district_given_rents(self, mu):
        """
        Phase 2: after rents are cleared, pick district maximizing overall utility.
        """
        expA, expB, expC = np.exp(self.U_A), np.exp(self.U_B), np.exp(self.U_C)
        total = expA + expB + expC
        Probs = {'a': expA/total, 'b': expB/total, 'c':expC/total}
        best_d, best_val = None, -float('inf')
        for district in self.model.districts:
            E_payoff = (Probs['a'] * (self.model.pay_c*mu[district]['a'] + self.model.pay_m*(1-mu[district]['a'])) + 
                        Probs['b'] * (self.model.pay_c*mu[district]['b'] + self.model.pay_m*(1-mu[district]['b'])) + 
                        Probs['c'] * (self.model.pay_c*mu[district]['c'] + self.model.pay_m*(1-mu[district]['c'])))

            # consumption with updated rent
            if self.model.alpha != 0:
                rent = district.rent
                consumption = max(self.income - min(rent, self.income), 0)
                utility = (consumption**self.model.alpha) * (E_payoff**(1-self.model.alpha))
            else:
                utility = E_payoff

            if utility > best_val:
                best_val, best_d = utility, district

        return best_d

    def move(self, best_district):
        """
        Relocate the agent to a new empty cell, preferring the given district.

        Args:
            best_district (District): The district whose expected behavior best matches.

        """
        if best_district.empty_places:
            new_x, new_y = self.random.choice(best_district.empty_places)
            current_district = self.model.district_of[self.pos]
            current_district.move_out(self)

            self.model.grid.move_agent(self, (new_x, new_y))
            new_district = self.model.district_of[self.pos]
            new_district.move_in(self)
            self.tenure = 0
            self.utility_sum = 0
        
        # if no empty cells in the preferred district, move to random district
        else:
            empties = list(self.model.grid.empties)
            if empties:
                new_pos = self.random.choice(empties)
                current_district = self.model.district_of[self.pos]
                current_district.move_out(self)

                self.model.grid.move_agent(self, new_pos)
                new_district = self.model.district_of[self.pos]
                new_district.move_in(self)
                self.tenure = 0
                self.utility_sum = 0
        return
    
    def step(self):
        
        # choose action and play coordination game with neighbors
        self.choose_action()
        mean_payoff = self.play_game()

        # update properties after collecting expected utilites
        self.update_utility_after_games(mean_payoff)
        self.utility_sum += mean_payoff
        self.tenure += 1

         # If the agent is unhappy and has reached the max tenure, the agent wants to move
        mean_utility = self.average_utility()

        self.happy = (mean_utility >= self.model.u_threshold)
        if self.happy:
            self.model.happiness_per_type[self.type] += 1

        unhappy_move = (mean_utility < self.model.u_threshold and self.tenure >= self.model.max_tenure)
        random_move = (self.tenure >= self.model.max_tenure and np.random.random() < self.model.p_random)
        
        if unhappy_move or random_move:
            best_district = self.choose_best_district_given_rents(self.model.mu)
            self.move(best_district)


import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation

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
        self.grid = MultiGrid(width, height, torus=False)

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

        #  metrics
        self.metrics = (self.exposure_to_others(self.num_agents_per_type, self.districts), 
                        self.dissimilarity(self.num_agents_per_type, self.districts)) 
        
        assert sum(self.num_agents_per_type) == self.schedule.get_agent_count(), "total number of agents across types is not the same as total agents"
        

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

            d.bids = []
            total_people += d.total_in_dist()
        
         # make sure the dynamical update results in same measures over time 
        assert total_people == self.schedule.get_agent_count(), f"Number of agents over districts ({total_people}) does not equal total number ({self.schedule.get_agent_count()})"



    def step(self):

        # reset happiness
        
        self.happy = 0
        self.happiness_per_type = [0.0, 0.0, 0.0]
        
        # Calculate action fractions
        self.calc_mu()

        # Only Calculate rents if alpha is nonzero
        if self.alpha > 0:
            # 1: Remove the bids from the last round
            for d in self.districts:
                d.bids.clear()

            # 2: Every agent submits its WTP for each district
            for agent in self.schedule.agents:
                agent.bid_for_districts(self.mu)

            # 3: Set the rent of each district
            self.set_rent_districts()

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

        


import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import numpy as np

def create_animation(snapshots):
    """
    Create and save an animation from a sequence of grid snapshots.

    Params:
        snapshots(list of np.ndarray): List of 2D arrays representing the grid state at each frame.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.subplots_adjust(top=0.85)   
    im = ax.imshow(snapshots[0], interpolation='nearest', origin='lower')
    ax.set_axis_off()
    fig.tight_layout()

    def update(frame):
        im.set_data(snapshots[frame])
        ax.set_title(f"Step {frame+10}")
        fig.tight_layout()
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), blit=True, interval=10)
    writer = animation.PillowWriter(fps=20)
    ani.save("schelling_animation.gif", 
             writer=writer)
    plt.close(fig)
    plt.show()




def happyness_plot(happy_data, happiness_grouped, numagents_per_type):
    """
    Create and save a plot showing the evolution of the proportion of happy agents.

    Params:
        happy_data(list or np.ndarray): Number of happy agents at each step.
        numagents(int): Total number of agents in the model.

    """

    output_path = "happiness_evolution.png"
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)

    
    fig, ax = plt.subplots(figsize=(5, 3))
    steps = np.arange(len(happy_data[0]))

    # turn into arrays
    arr_tot = np.stack(happy_data)            # shape = (n_runs, T)
    arr_grp = np.stack(happiness_grouped)     # shape = (n_runs, T, n_types)
    na      = np.stack(numagents_per_type)    # shape = (n_runs, n_types)
    
    # per-run totals
    total_agents = na.sum(axis=1)             # (n_runs,)

    # proportions
    prop_tot = arr_tot / total_agents[:,None] # (n_runs, T)
    prop_grp = arr_grp / na[:,None,:]         # (n_runs, T, n_types)

    mean_tot = prop_tot.mean(axis=0)                
    var_tot  = prop_tot.var(axis=0)
    mean_grp = prop_grp.mean(axis=0)               
    var_grp  = prop_grp.var(axis=0)

    ax.plot(steps, mean_tot, label="Total", linewidth=0.3)
    ax.fill_between(steps,
                    mean_tot - np.sqrt(var_tot),
                    mean_tot + np.sqrt(var_tot),
                    alpha=0.2)
    
    for i in range(3):
        mu = mean_grp[:, i]
        sigma = np.sqrt(var_grp[:, i])
        ax.plot(steps, mu, label=f"Type {i}", linewidth=0.3)
        ax.fill_between(steps, mu - sigma, mu + sigma, alpha=0.2)
            
    ax.set_xlabel("steps")
    ax.set_ylabel("proportion happy")
    ax.set_title("development of happyness")
    fig.tight_layout()
    plt.legend()
    fig.savefig(output_path)
    plt.close(fig)



def metrics_plot(metrics_diss_exp):
    """
    Create and save a plot showing the evolution of the proportion of happy agents.

    Params:
        happy_data(list or np.ndarray): Number of happy agents at each step.
        numagents(int): Total number of agents in the model.

    Returns: (None): The plot is saved to 'plots/happyness_evolution.png'.
    """

    output_path = "metrics_evolution.png"
    metrics_arr = np.array(metrics_diss_exp)
    
    # Compute mean and variance across runs (axis=0) → shape (steps, 2)
    mean_vals = metrics_arr.mean(axis=0)
    var_vals  = metrics_arr.var(axis=0)
    
    steps = np.arange(mean_vals.shape[0])

    fig, ax = plt.subplots(figsize=(5, 3))

    # Dissimilarity (metric index 0)
    mean_diss   = mean_vals[:, 1]
    sigma_diss  = 1.96 * np.sqrt(var_vals[:, 1]) / np.sqrt(10)
    ax.plot(steps, mean_diss, label="Dissimilarity")
    ax.fill_between(steps,
                    mean_diss - sigma_diss,
                    mean_diss + sigma_diss,
                    alpha=0.2)
    
    # Exposure (metric index 1)
    mean_exp    = mean_vals[:, 0]
    sigma_exp   = 1.96* np.sqrt(var_vals[:, 0]) / np.sqrt(10)
    ax.plot(steps, mean_exp, label="Exposure")
    ax.fill_between(steps,
                    mean_exp - sigma_exp,
                    mean_exp + sigma_exp,
                    alpha=0.2)
    
    ax.set_xlabel("Steps")
    ax.set_ylabel("Metric value")
    ax.set_title("Evolution of Dissimilarity and Exposure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

import numpy as np


from concurrent.futures import ProcessPoolExecutor
from functools import partial

# -------------------------- Simulation Setup ----------------------------- #
def run_model(which_run, steps, seedje, params):
    """
    Run the given model for a fixed number of steps, recording the grid state at each step.

    Params
    steps(int): Number of iterations to advance the model.
    model( mesa object):  Model instance providing a `.grid.coord_iter()` method and a `.step()` method.

    Returns: (List[np.ndarray]): A list of 2D arrays capturing the grid at each step, where
        empty cells are marked as –1 and occupied cells by agent type.
    """
    snapshots = []
    happyness = []
    (width, height, density, p_random, pay_c, pay_m, max_tenure, u_threshold, alpha, pop_distribution) = params
    model = SchellingModel(width, height, density, p_random, pay_c, pay_m, max_tenure, u_threshold, alpha, pop_distribution, seedje+which_run)
    run_metrics = []

    for s in range(steps):
        # Capture grid state: -1 empty, 0, 1 or 2 agent types
        grid_state = np.full((width, height), -1)
        for cell in model.grid.coord_iter():
            content, x, y = cell
            if content:

                # capacity 1 grid, so take first agent
                grid_state[x, y] = content[0].type
        if s%10 == 0:
            snapshots.append(grid_state)
        happyness.append(model.happiness_per_type)
        run_metrics.append(model.metrics)
        # if model.happy < model.schedule.get_agent_count(): 
        model.step()
        # else: 
        #     break
    print(f"happiness is reached after {s} steps.")
    # print(run_metrics)
    return snapshots, happyness, run_metrics, model.num_agents_per_type,  which_run


def execute_parallel_models(how_many, 
                            params,
                            seedje, 
                            steps):
    """
    Run multiple Schelling model simulations in parallel.

    Args:
        how_many (int): Number of runs to execute.
        params (dict): Model parameters.
        seedje (int): Base random seed.
        steps (int): Number of steps per run.

    Returns:
        list: Results from each parallel run.
    """

    print(f"starting parallel generation of Schelling model for {how_many} runs)")
    print("-----------------------------------------")
    runs = np.arange(how_many)  # Create a range for the runs
    num_threads = min(how_many, 10)
    # Partially apply parameters for the worker function (currently one parameter setting, could vary with more)
    worker_function = partial(
        run_model,
        steps = steps,
        seedje = seedje,
        params = params
    )
        
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(worker_function, runs))
        
    return results


def unpack_and_parse_results(resultaatjes):
    """
    Align and pad runs to equal length, then compute total happiness per step.

    Params:
        resultaatjes (list of tuples):  
            Each element is (snapshots, happiness_measures, run_metrics,
            num_agents_grouped, run_id) for one run.

    Returns:
        tuple:
            snapshots_runs (Tuple): of snapshots per run;
            happiness_runs (List): padded happiness lists per run;
            run_metrics (List): padded metrics lists per run;
            num_agents_grouped (Tuple): agent counts per run.
    """
    # unpack results
    snapshots_runs, happiness_runs, run_metrics, num_agents_grouped, run_ids = zip(*resultaatjes)
    total_happy = []
    maximal_runs = max([len(x) for x in happiness_runs])

    # iterate through different runs
    for i, happiness_measures in enumerate(happiness_runs): 

        # if this is not the longest run, padd with last value
        if len(happiness_measures) < maximal_runs:
            diff = maximal_runs - len(happiness_measures)
            last_happy = happiness_measures[-1]
            last_metric = run_metrics[i][-1]
            happiness_measures.extend([last_happy] * diff)
            run_metrics[i].extend([last_metric] * diff)
        
        # compute total happiness (over all types)
        tot_happyness = np.sum(happiness_measures, axis=1)
        total_happy.append(tot_happyness)
    
    return snapshots_runs, happiness_runs, total_happy, run_metrics, num_agents_grouped
        


def main():

    # set parameters 
    width = 10
    height = 10
    density = 0.9

    # based on real data
    population_distribution = [0.1752,0.524,0.3008]
    
    p_random = 0.1
    pay_c = 10
    pay_m = 4
    min_tenure = 5
    u_threshold = 8

    alpha = 0.5

    steps = 2000
    seedje = 43
    num_runs = 10

    # pack params
    params = (width, height, density, 
              p_random, pay_c, pay_m,
              min_tenure, u_threshold, alpha,
              population_distribution)
    
    # run singular model
    # run_model(0, steps, seedje, params)
    
    # parallel
    # collect all results over runs (parallelized implementation)
    resultaatjes = execute_parallel_models(num_runs,
                                           params,
                                           seedje,
                                           steps)
    

    # collect data of run 
    snapshots_runs, happiness_runs, total_happy, run_metrics,num_agents_grouped = unpack_and_parse_results(resultaatjes)
    
    # visualize
    happyness_plot(total_happy, happiness_runs, num_agents_grouped)
    metrics_plot(run_metrics)
    create_animation(snapshots_runs[0])

if __name__ == '__main__':
    main()

