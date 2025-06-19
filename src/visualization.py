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




def happyness_plot(happy_data, happiness_grouped, numagents_per_type):
    """
    Create and save a plot showing the evolution of the proportion of happy agents.

    Params:
        happy_data(list or np.ndarray): Number of happy agents at each step.
        numagents(int): Total number of agents in the model.

    """

    output_path = "plots/happiness_evolution.png"
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

    ax.plot(steps, mean_tot, label="Total")
    ax.fill_between(steps,
                    mean_tot - np.sqrt(var_tot),
                    mean_tot + np.sqrt(var_tot),
                    alpha=0.2)
    for i in range(3):
        mu = mean_grp[:, i]
        sigma = np.sqrt(var_grp[:, i])
        ax.plot(steps, mu, label=f"Type {i}")
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

    output_path = "plots/metrics_evolution.png"
    metrics_arr = np.array(metrics_diss_exp)
    
    # Compute mean and variance across runs (axis=0) → shape (steps, 2)
    mean_vals = metrics_arr.mean(axis=0)
    var_vals  = metrics_arr.var(axis=0)
    
    steps = np.arange(mean_vals.shape[0])

    fig, ax = plt.subplots(figsize=(5, 3))

    # Dissimilarity (metric index 0)
    mean_diss   = mean_vals[:, 1]
    sigma_diss  = np.sqrt(var_vals[:, 1])
    ax.plot(steps, mean_diss, label="Dissimilarity")
    ax.fill_between(steps,
                    mean_diss - sigma_diss,
                    mean_diss + sigma_diss,
                    alpha=0.2)
    
    # Exposure (metric index 1)
    mean_exp    = mean_vals[:, 0]
    sigma_exp   = np.sqrt(var_vals[:, 0])
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