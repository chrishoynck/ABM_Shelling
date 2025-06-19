import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation


# ---------------------- Model and Agent Definitions ---------------------- #

class SchellingAgent(Agent):
    """Simple Schelling segregation agent."""
    def __init__(self, unique_id, model, agent_group):
        super().__init__(unique_id, model)
        self.type = agent_group
        self.U_A = self.model.pay_c if agent_group==1 else self.model.pay_m # A = 1
        self.U_B = self.model.pay_c if agent_group==2 else self.model.pay_m # B = 2
        self.U_C = self.model.pay_c if agent_group==3 else self.model.pay_m # C = 3
        self.N_A = self.N_B = self.N_C = 10
        self.current_action = np.random.choice(['a', 'b', 'c'])
        self.tenure = 0
        self.utility_sum = 0
        self.radius = 1
        self.happy = None

    def choose_action(self):
        exp_A = np.exp(self.U_A)
        exp_B = np.exp(self.U_B)
        exp_C = np.exp(self.U_C)
        total = exp_A + exp_B + exp_C

        p_A = exp_A / total
        p_B = exp_B / total
        p_C = exp_C / total
        self.current_action = np.random.choice(['a', 'b', 'c'], p=[p_A, p_B, p_C])

    def update_utility_after_games(self, new_payoff):
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

    def step(self):

        # Choose an action for each agent according to the Logit model
        self.choose_action()

        # Obtain the neighbors of the agent
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=self.radius)
        # Play the coordination game with each neighbour
        total_payoff_games = 0
        if neighbors:
            for n in neighbors:
                if self.current_action == n.current_action:
                    payoff = self.model.pay_c
                else:
                    payoff = self.model.pay_m
                total_payoff_games += payoff
        
        mean_payoff = total_payoff_games / len(neighbors) if len(neighbors) != 0 else 0
        self.update_utility_after_games(mean_payoff)
        self.utility_sum += mean_payoff
        self.tenure += 1

        # If the agent is unhappy and has reached the max tenure, the agent wants to move
        mean_utility = self.average_utility()
        unhappy_move = (mean_utility < self.model.u_threshold and self.tenure >= self.model.max_tenure)
        if not unhappy_move:
            self.happy = True

        random_move = (self.tenure >= self.model.max_tenure and random.random() < self.model.p_random)
        if unhappy_move or random_move:
            self.happy = False
            counts_per_district = {
                district: {'a':0, 'b':0, 'c':0, 'total':0}
                for district in (0, 1, 2)
            }

            # For each district, counts the number of agents playing a, b, and c
            for agent in self.model.schedule.agents:
                x, y = agent.pos
                district = 3*x // self.model.width
                counts_per_district[district][agent.current_action] += 1
                counts_per_district[district]['total'] += 1
            
            # Calculate mu for each district
            mu = {}
            for district, counts in counts_per_district.items():
                mu[district] = {
                    'a': counts['a'] / counts['total'],
                    'b': counts['b'] / counts['total'],
                    'c': counts['c'] / counts['total']
                }

            # Calculate expected payoff for each district
            expA, expB, expC = np.exp(self.U_A), np.exp(self.U_B), np.exp(self.U_C)
            total = expA + expB + expC
            Probs = {'a': expA/total, 'b': expB/total, 'c':expC/total}
            E_payoffs_per_district = {}
            for district in (0, 1, 2):
                E_payoffs_per_district[district] = (Probs['a'] * (self.model.pay_c*mu[district]['a'] + self.model.pay_m*(1-mu[district]['a'])) + 
                                                    Probs['b'] * (self.model.pay_c*mu[district]['b'] + self.model.pay_m*(1-mu[district]['b'])) + 
                                                    Probs['c'] * (self.model.pay_c*mu[district]['c'] + self.model.pay_m*(1-mu[district]['c'])))
            best_district = max(E_payoffs_per_district, key=E_payoffs_per_district.get)

            # Find an empty cell in the preferred district and place the agent there
            empty_cells = []
            x_min = best_district * (self.model.width // 3)
            x_max = x_min + (self.model.width // 3)
            for x in range(x_min, x_max):
                for y in range(self.model.height):
                    if self.model.grid.is_cell_empty((x, y)):
                        empty_cells.append((x, y))
            
            if empty_cells:
                new_x, new_y = random.choice(empty_cells)
                self.model.grid.move_agent(self, (new_x, new_y))
                self.tenure = 0
                self.utility_sum = 0
        

class SchellingModel(Model):
    """Model class for the Schelling segregation model."""
    def __init__(self, width, height, density, p_random, pay_c, pay_m, max_tenure, u_threshold, population_distribution=[1/3, 1/3, 1/3], seed=None):
        super().__init__(seed)
        self.width = width
        self.height = height
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, torus=False)
        self.pay_c = pay_c
        self.pay_m = pay_m
        self.max_tenure = max_tenure
        self.u_threshold = u_threshold
        self.p_random = p_random

        # Determine the number of Agents in groups A, B, and C
        frac_low, frac_medium, frac_high = population_distribution
        num_agents = int(density * width * height)
        counts_per_group = {
            'A': int(num_agents * frac_low),
            'B': int(num_agents * frac_medium),
            'C': num_agents - int(num_agents * frac_low) - int(num_agents * frac_medium)
        }

        # Give each Agent a random position within the grid for initialisation
        possible_positions = [(x, y) for x in range(width) for y in range(height)]
        choose_random_positions = random.sample(possible_positions, num_agents)
        uid = 0

        # Place all Agents of each type on the grid
        for i in range(counts_per_group['A']):
            x, y = choose_random_positions[uid]
            A_agent = SchellingAgent(uid, self, 1)
            self.grid.place_agent(A_agent, (x, y))
            self.schedule.add(A_agent)
            uid += 1

        for j in range(counts_per_group['B']):
            x, y = choose_random_positions[uid]
            B_agent = SchellingAgent(uid, self, 2)
            self.grid.place_agent(B_agent, (x, y))
            self.schedule.add(B_agent)
            uid += 1

        for k in range(counts_per_group['C']):
            x, y = choose_random_positions[uid]
            C_agent = SchellingAgent(uid, self, 3)
            self.grid.place_agent(C_agent, (x, y))
            self.schedule.add(C_agent)
            uid += 1

    def step(self):
        self.schedule.step()


# --------------------------- Simulation Setup ----------------------------- #

# Simulation parameters
width = 15
height = 15
density = 0.9
p_random = 0.1
pay_c = 10
pay_m = 7
max_tenure = 5
u_threshold = 7
population_distribution = [0.25, 0.5, 0.25]
radius = 1
steps = 1000

# Initialize and run model
model = SchellingModel(width, height, density, p_random, pay_c, pay_m, max_tenure, u_threshold, population_distribution)
snapshots = []
happiness_fractions = []

for _ in range(steps):
    grid_state = np.full((width, height), -1)
    for cell in model.grid.coord_iter():
        content, x, y = cell
        if content:
            # capacity 1 grid, so take first agent
            grid_state[x, y] = content[0].type
    snapshots.append(grid_state)
    happiness_fractions.append(sum(1 for agent in model.schedule.agents if agent.happy) / len(model.schedule.agents))
    model.step()

# --------------------------- Create Animation ----------------------------- #

# plt.plot(happiness_fractions)
# plt.show()



fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(snapshots[0], interpolation='nearest')
ax.set_axis_off()
fig.tight_layout()

def update(frame):
    im.set_data(snapshots[frame])
    ax.set_title(f"Step {frame+1}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(snapshots), blit=True, interval=100)
plt.show()

# plt.figure(figsize=(15,15))

# for i in range(25):

#     plt.subplot(5, 5, i+1)
#     plt.title(f'step {i}')
#     plt.imshow(snapshots[i])

#     plt.axis('off')
# plt.show()

# plt.figure(figsize=(20, 5))

# for i, step in enumerate([15, 50, 500, 1400]):
#     plt.subplot(1, 4, i+1)
#     plt.title(f'step {step}')
#     plt.imshow(snapshots[step])

#     plt.axis('off')
# plt.show()

