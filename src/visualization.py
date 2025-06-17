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
    im = ax.imshow(snapshots[0], interpolation='nearest')
    ax.set_axis_off()
    fig.tight_layout()

    def update(frame):
        im.set_data(snapshots[frame])
        ax.set_title(f"Step {frame+1}")
        fig.tight_layout()
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), blit=True, interval=100)
    writer = animation.PillowWriter(fps=10)
    ani.save("plots/schelling_animation.gif", 
             writer=writer)
    plt.close(fig)
    plt.show()


def happyness_plot(happy_data, numagents):
    """
    Create and save a plot showing the evolution of the proportion of happy agents.

    Params:
        happy_data(list or np.ndarray): Number of happy agents at each step.
        numagents(int): Total number of agents in the model.

    Returns: (None): The plot is saved to 'plots/happyness_evolution.png'.
    """

    output_path = "plots/happyness_evolution.png"
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 3))
    steps = np.arange(len(happy_data))
    ax.plot(
        steps,
        np.array(happy_data) / numagents,
        marker='o',
        linestyle='-',
        markersize=4
    )
    ax.set_xlabel("steps")
    ax.set_ylabel("proportion happy")
    ax.set_title("development of happyness")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)