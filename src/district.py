import numpy as np

class District:
    def __init__(self, id_):
        """
        Initialize a District.

        Params:
            id_ (int): Unique district identifier.
            rent (float): initial_rent 
        """

        self.id = id_

        # counts of agent types within district 
        self.counts = {0: 0, 1: 0, 2: 0}

        # counts of played actions within district
        self.action_counts = {'a': 0, 'b': 0, 'c': 0}

        # total space and empty space of district 
        self.empty_places = []
        self.area = 0

        # Biddings on district (all bids and only hgihest bids)
        self.bids = []
        self.high_bids = []

        # rents 
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
        """
        Adjust the district's rent based on high bids relative to its supply and a density factor, 
        then clear all recorded bids.

        Raises:
            ValueError: If there are no bids when attempting to decrease rent.
        """

        # supply is equal to the area of the district. 
        supply = self.area
        rent = self.rent

        #  if number of highest bids exceeds the supply
        if len(self.high_bids) >= supply:
            rent =  1.01* self.next_rent 

        # if supply is less than the number of bids (minus small density factor)
        elif len(self.high_bids) < supply*(1-self.id/5):
            rent = 0.99*self.next_rent 
            if not self.bids:
                raise ValueError("there are no biddings on this district, should not happen")
        
        # update rents, next rent is set to rent, current rent is set to next rent
        self.next_rent = rent
        self.rent = self.next_rent
        self.bids = []
        self.high_bids = []

    