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

        base_income = {0: 4, 1: 10, 2: 16}[agent_type]
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

    def get_best_district(self, mu): 
        """
        Select the district with highest expected payoff.
        Places bids on houses within districts to determine rent for next round

        Params:
            mu (dict): Mapping each District to a dict of action proportions.

        Returns:
            District: The district maximizing the weighted expected payoff.
        """
        expA, expB, expC = np.exp(self.U_A), np.exp(self.U_B), np.exp(self.U_C)
        total = expA + expB + expC
        Probs = {'a': expA/total, 'b': expB/total, 'c':expC/total}
        E_payoffs_per_district = {}
        for district in self.model.districts:
            E_payoff = (Probs['a'] * (self.model.pay_c*mu[district]['a'] + self.model.pay_m*(1-mu[district]['a'])) + 
                        Probs['b'] * (self.model.pay_c*mu[district]['b'] + self.model.pay_m*(1-mu[district]['b'])) + 
                        Probs['c'] * (self.model.pay_c*mu[district]['c'] + self.model.pay_m*(1-mu[district]['c'])))
            
            # my_district = self.model.district_of[self.pos]
            rent = district.rent
            consumption = self.income - min(rent, self.income)

            # WTP = (consumption**self.model.alpha) * (E_payoff**(1-self.model.alpha))

            # WITHOUT HOUSING RENTS 
            WTP = E_payoff

            
            district.bids.append((self, WTP))
            E_payoffs_per_district[district] =  WTP #(consumption**self.model.alpha) * (E_payoff**(1-self.model.alpha))
            # print(f"agent type: {self.type}, income: {np.round(self.income, 2)}, district: {district.id}, WTP: {np.round(WTP, 2)}, consumption: {np.round(consumption, 2)}")
        best_district = max(E_payoffs_per_district, key=E_payoffs_per_district.get)

        return best_district

    def move(self, best_district, random_move):
        """
        Relocate the agent to a new empty cell, preferring the given district.

        Args:
            best_district (District): The district whose expected behavior best matches.

        """

        # change locations and districts 
        if not random_move and np.random.random() < 0.3:
            return
        if best_district.empty_places and not random_move:
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

        # move if not happy or with random small prob
        random_move = (self.tenure >= self.model.max_tenure and np.random.random() < self.model.p_random)
        
        best_district = self.get_best_district(self.model.mu)

        if unhappy_move or random_move:
            self.move(best_district, random_move)


        
                
