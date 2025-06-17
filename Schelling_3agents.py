import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.model import SchellingModel

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
