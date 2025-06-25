import numpy as np

class District:
    def __init__(self, id_):
        """
        Initialize a District.

        Params:
            id_ (int): Unique district identifier.
            rent (float): initial_rent 
        """
        # super().__init__(model)
        self.id = id_
        self.counts = {0: 0, 1: 0, 2: 0}
        self.action_counts = {'a': 0, 'b': 0, 'c': 0}
        self.empty_places = []
        self.bids = []
        self.area = 0
        self.min = 0
        self.max = 0
        self.high_bids = []
        self.next_rent = 0
        self.rent = 0

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


    def my_rent_based_on_bids(self):
        # self.high_bids.sort(key=lambda x: x[1], reverse=True)
        # values = [bid for (_, bid) in self.bids]
        supply = self.area
        
        # print(len(d.bids))
        rent = self.rent
        if len(self.high_bids) >= supply:
            rent =  1.01* self.next_rent #self.high_bids[supply-2][1] #np.mean(values)
        elif len(self.high_bids) < supply*(1-self.id/5):
            rent = 0.99*self.next_rent # self.bids[-1][1] if self.high_bids else 0  
            if not self.bids:
                raise ValueError("there are no biddings on this district, should not happen")
        # self.next_rent = (np.average([rent, self.rent], weights=[0.1, 0.9]))
        self.next_rent = rent
        self.rent = self.next_rent
        self.bids = []

    # def my_rent_based_on_bids(self):
    #     # self.high_bids.sort(key=lambda x: x[1], reverse=True)
    #     values = [bid for (_, bid) in self.bids]
    #     supply = self.area
        
    #     # print(len(d.bids))
    #     if len(self.high_bids) >= supply:
    #         rent = np.mean(values)
    #     else:
    #         rent = 0.9*self.next_rent # self.bids[-1][1] if self.high_bids else 0  
    #         if not self.bids:
    #             raise ValueError("there are no biddings on this district, should not happen")
    #     self.next_rent = (np.average([rent, self.rent], weights=[0.1, 0.9]))
    #     self.rent = self.next_rent
    #     self.bids = []

    #     # sort high_bids by WTP descending
    #     self.high_bids.sort(key=lambda x: x[1], reverse=True)
    #     self.bids.sort(key=lambda x: x[1], reverse=True)

    #     if len(self.high_bids) > self.area:
    #         # uniform price among first-choice bidders
    #         rent = self.high_bids[self.area][1]
    #     else:
    #         # under-demand: maybe set rent to the lowest first-choice bid 
    #         if self.high_bids:
    #             rent = self.high_bids[-1][1]
    #         else:
    #             # print(f"No one wants to live here {self.id}, my rent is: {self.rent}")
    #             # rent = 0
    #             rent = self.bids[-1][1]
    #         self.rent = self.next_rent
    #         self.next_rent = (np.average([rent, self.rent], weights=[0.1/(1+0.1), 1/(1/(1+0.1)) ]))  
    
    # def my_rent_based_on_bids(self):

    #     self.bids.sort(key=lambda x: x[1], reverse=True)
    #     # self.high_bids.sort(key=lambda x: x[1], reverse=True)

    #     supply = self.area
    #     if supply == 0:
    #         return 
        
    #     values = [bid for (_, bid) in self.high_bids]
    #     indexje = supply # int((len(self.empty_places)/supply) *(20*20))
    #     # print(len(d.bids))
    #     if len(self.high_bids) > indexje:
    #         rent = 1.1*np.mean(values)
    #     elif self.high_bids:
    #         rent = 0.9* np.mean(values)
    #         if not self.bids:
    #             raise ValueError("There are no biddings on this district, should not happen")
    #     else: 
    #         rent = self.min
    #     rent = min(max(self.min, rent), self.max)
    #     self.rent = self.next_rent
    #     self.next_rent = rent

    #     # if less highest bids have come in then open places (nobody's first choice)
    #     # if (len(self.high_bids)) < len(self.empty_places): 
    #     #     # print(f"No one wants to live here {self.id}")
    #     #     rent*=0.9

    #     self.next_rent = (np.average([rent, self.rent], weights=[0.01, 0.99]))
        
        

    #     # empty the bids 
    #     self.bids = []
    #     self.high_bids = []


    # def my_rent_based_on_bids(self):

    #     self.bids.sort(key=lambda x: x[1], reverse=True)
    #     supply = self.area
    #     total_area = 20*20

    #     if self.area == 0: 
    #         return 
        
    #     area_frac  = self.area / total_area

    #     # if self.area> 0:
    #     #     empty_frac = len(self.empty_places)/ self.area
    #     #     index = int(empty_frac*total_area)
    #     #     rent = self.bids[len(self.empty_places)][1]
    #     #     self.rent = self.next_rent
    #     #     self.next_rent = (np.average([rent, self.rent], weights=[0.1, 0.9]))

    #     factor = 1
    #     if len(self.high_bids) > (supply): 
    #         factor = 1 + area_frac/10
    #     elif len(self.high_bids) < (supply):
    #         factor = 1 - area_frac/10
    #     self.rent = self.next_rent
    #     self.next_rent = min(max(factor*self.rent, self.min), self.max)
        

    #     # # advance to next period
    #     # self.rent = self.next_rent
    #     self.bids = []
    #     self.high_bids = []
