import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation


# ---------------------- Model and Agent Definitions ---------------------- #

class SchellingAgent(Agent):
    """Simple Schelling segregation agent."""
    def __init__(self, unique_id, model, agent_type, homophily=0.4, radius=1):
        super().__init__(unique_id, model)
        self.type = agent_type
        self.homophily = homophily
        self.radius = radius
        self.happy = False

    def step(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=self.radius
        )
        if neighbors:
            similar = sum(1 for n in neighbors if n.type == self.type)
            frac = similar / len(neighbors)
        else:
            frac = 0
        self.happy = frac >= self.homophily
        if not self.happy:
            empties = list(self.model.grid.empties)
            if empties:
                new_pos = self.random.choice(empties)
                self.model.grid.move_agent(self, new_pos)

class SchellingModel(Model):
    """Model class for the Schelling segregation model."""
    def __init__(self, width, height, density, income_distribution, homophilies, radius, seed=None):
        super().__init__(seed)
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, torus=True)
        uid = 0
        assert sum(income_distribution) == 1, "Sum of income distribution should be 1"
        homophily_low, homophily_medium, homophily_high = homophilies
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

    def step(self):
        self.schedule.step()


# --------------------------- Simulation Setup ----------------------------- #

# Simulation parameters
width = 20
height = 20
density = 0.9
income_distribution = [1/3, 1/3, 1/3]
homophilies = [0.3, 0.3, 0.3]
radius = 1
steps = 100

# Initialize and run model
model = SchellingModel(width, height, density, income_distribution, homophilies, radius)
snapshots = []

for _ in range(steps):
    # Capture grid state: -1 empty, 0 or 1 agent types
    grid_state = np.full((width, height), -1)
    for cell in model.grid.coord_iter():
        content, x, y = cell
        if content:
            # capacity 1 grid, so take first agent
            grid_state[x, y] = content[0].type
    snapshots.append(grid_state)
    model.step()

# --------------------------- Create Animation ----------------------------- #

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
