from mesa import Agent
import numpy as np

# class changed with coordination game
class SchellingAgent(Agent):
    def __init__(self, unique_id, model, agent_type, radius=1):
        super().__init__(unique_id, model)
        self.type = agent_type
        self.radius = radius

    def step(self):
        # choose action
        self.action = choose_action(self.prop, self.random)

        # play coordination game
        play_coordination_game(self, self.prop, self.count,
                       self.model.grid, self.radius)

        # relocate if unhappy
        if self.payoff < self.threshold:
            empties = list(self.model.grid.empties)
            if empties:
                new_pos = self.random.choice(empties)
                self.model.grid.move_agent(self, new_pos)

# Parameters
actions = ("a", "b", "c") # 3 actions here
pi_c, pi_m = 10, 1 # pay-offs
threshold = 8 # threshold for happinness

# Choice rule for action
def choose_action(prop):
    """
    prop: dictionary for action → propensity
    returns: one of "a", "b", or "c" drawn with soft-max probabilities
    """

    # numerator for each action:
    num = {
        a: np.exp(prop[a])             # e^{ĥπ_it(a)}
        for a in actions
    }

    # denominator is the sum over all numerators:
    den = sum(num.values())       # e^{ĥπ_it(a)} + e^{ĥπ_it(b)} + e^{ĥπ_it(c)}

    # probabilities:
    probs = {
        a: num[a] / den            # e^{ĥπ_it(a)} / (sum of e^{)
        for a in actions
    }

    # sample according probs and random
    return np.random.choice(
        actions,
        p=[probs[a] for a in actions]
    )

# Play game
def play_coordination_game(agent, prop, count, grid, radius):

    # collect neighbors N_it in a list
    neighs = grid.get_neighbors(
        agent.pos,
        moore=True,
        include_center=False,
        radius=radius
    )

    # compute pi_it = payoff
    if neighs: # if there are neighbours:
        k = sum(1 for n in neighs if n.action == agent.action) # k = number of neighbors whose action matches

        # total payoff from matching neighbors: each gives pi_c
        # total payoff from mismatching neighbors: each gives pi_m
        # k * pi_c = sum of payoffs from the k matches
        # (len(neighs)-k) * pi_m = sum of payoffs from the mismatches
        # divide by len(neighs) to get the average payoff pi_it
        pi_it = (k * pi_c + (len(neighs) - k) * pi_m) / len(neighs)

    else: # if there are no neighbours: payoff stays same
        pi_it = prop[agent.action]

    # store for relocation
    agent.payoff = pi_it

    # update propensity for action chosen
    a = agent.action
    c = count[a]
    prop[a] = (c * prop[a] + pi_it) / (c + 1)
    count[a] = c + 1