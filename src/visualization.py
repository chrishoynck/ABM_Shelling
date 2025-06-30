import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np


COLOURS = ["#dfa83b", "#a26da4", "#496636"]
COLOUR_MET = ["#DF7F31", "#1F644E"]
def create_animation(snapshots, alpha, save):
    """

    Create and save an animation from a sequence of grid snapshots.

    Params:
        snapshots(list of np.ndarray): List of 2D arrays representing the grid state at each frame.
    """

    # build a ListedColormap
    cmap = ListedColormap(["white"]+ COLOURS)

    # define bin edges so that 0→bin[0], 1→bin[1], 2→bin[2]
    norm = BoundaryNorm(boundaries=[-1 , 0,1,2, 3], ncolors=cmap.N)

    # save snapshots
    for step in (0, 10, 200):
        fig2, ax2 = plt.subplots(figsize=(5,5))
        stepje = step
        if step > 0: 
            stepje = step-1
        img = np.rot90(snapshots[stepje], k=1)
        h, w = img.shape

        ax2.imshow(img, interpolation='nearest', origin='lower',
                cmap=cmap, norm=norm)
        ax2.set_xticks([])
        ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('white')
            spine.set_linewidth(20)
            spine.set_position(('outward', 20)) 

        ax2.set_title(f"Step {step*10}", fontsize=35)
        fig2.tight_layout(pad=0)
        fig2.savefig(f"plots/animations/snapshots/{alpha}/step_{step}.png", dpi=300,
                    bbox_inches='tight', pad_inches=0.2)
        plt.close(fig2)

    # create animation
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.subplots_adjust(top=0.85)   
    im = ax.imshow(snapshots[0], interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
    ax.set_axis_off()
    fig.tight_layout()

    def update(frame):
        data = np.rot90(snapshots[frame], k=1)
        im.set_data(data)
        ax.set_title(f"Step {frame*10}")
        fig.tight_layout()
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), blit=True, interval=10)
    writer = animation.PillowWriter(fps=20)
    if save:
        ani.save(f"plots/animations/{alpha}.gif", 
                writer=writer)
    plt.close(fig)
    plt.show()


def occupants_per_dist(occ, alpha, number_of_places_per_district, save, output_path="plots/occupants_per_dist.png"):
    """
    Plot a grouped bar chart of average agent proportions per district for each agent type.

    Params:
        occ (np.ndarray): Array of shape (R, D, T) where
                          R = number of runs,
                          D = number of districts,
                          T = number of agent types.
        alpha (float):    Alpha parameter value (for title/filename).
        number_of_places_per_district (list or array): length-D list of total places per district.
        output_path (str): Path to save the figure as PNG.
    """
    # Compute mean and standard deviation across runs (axis=0)
    mean_counts = occ.mean(axis=0)          # shape (D, T)
    std_counts  = np.sqrt(occ.var(axis=0))  # shape (D, T)

    # Convert counts to proportions by dividing by district capacities
    capacities = np.array(number_of_places_per_district).mean(axis=0)  # shape (D,)

    # calculate proportions
    mean_props = mean_counts / capacities[:, None]
    std_props  = std_counts  / capacities[:, None]
    D, T = mean_props.shape
    indices = np.arange(D)
    width = 0.8 / T

    fig, ax = plt.subplots(figsize=(4, 3))

    for t in range(T):
        ax.bar(indices + t * width,
               mean_props[:, t],
               width=width,
               yerr=(std_props[:, t]*1.96)/np.sqrt(len(mean_counts)),
               capsize=3,
               color = COLOURS[t],
               label=f"Type {t}")

    ax.set_xticks(indices + width * (T - 1) / 2)
    ax.set_xticklabels([f"District {d}" for d in range(D)])
    ax.set_xlabel("District")
    ax.set_ylabel("Average proportion occupied")
    ax.set_title(r"Average Occupancy Proportions by District ( $\alpha$=" + f"{alpha}" + ")")
    ax.legend()
    fig.tight_layout()

    if save:
        fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return mean_props, std_props


def multiple_occ_per_dist(dict_occ, runs, save):
    fig, axs = plt.subplots(2, 2, figsize=(5,5), sharex=True, sharey=True)
    counter = 0
    axs = axs.flatten()
    for alpha, (mean_props, std_props) in dict_occ.items():

        D, T = mean_props.shape
        indices = np.arange(D)
        width = 0.8 / T

        # if alpha%0.2 == 0 and alpha < 1: 
        for t in range(T):
            axs[counter].bar(indices + t * width,
                mean_props[:, t],
                width=width,
                yerr=(std_props[:, t]*1.96)/np.sqrt(runs),
                color = COLOURS[t],
                capsize=3,
                label=f"Type {t}")
        
        axs[counter].set_title(r"$\alpha: $" + f"{alpha}")

        axs[counter].set_xticks(indices + width * (T - 1) / 2)
        axs[counter].set_xticklabels([f"{d}" for d in range(D)])
    
        if counter > 1:
            axs[counter].set_xlabel("District")

        if counter%2 == 0:
            axs[counter].set_ylabel("prop. occupied")
        
        if counter == 3:
            axs[counter].legend()
        counter +=1

        fig.suptitle(" Occupancy Proportions by District")
        fig.tight_layout()
        if save:
            fig.savefig("plots/occupants_suplots.png", dpi=300)
        plt.close(fig)
        
            
        

def happyness_plot(happy_data, happiness_grouped, numagents_per_type, save):
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

    ax.plot(steps, mean_tot, label="Total", linewidth=0.8, color = 'darkblue')
    ax.fill_between(steps,
                    mean_tot - np.sqrt(var_tot),
                    mean_tot + np.sqrt(var_tot),
                    alpha=0.2, color= 'blue')
    
    for i in range(3):
        mu = mean_grp[:, i]
        sigma = (np.sqrt(var_grp[:, i])*1.96)/np.sqrt(len(arr_tot))
        ax.plot(steps, mu, label=f"Type {i}", linewidth=0.8, color = COLOURS[i])
        ax.fill_between(steps, mu - sigma, mu + sigma, alpha=0.2, color = COLOURS[i])
            
    ax.set_xlabel("steps")
    ax.set_ylabel("proportion happy")
    ax.set_title("development of happyness")
    fig.tight_layout()
    plt.legend()
    if save:
        fig.savefig(output_path, dpi=300)
    plt.close(fig)

def district_prices_plot(district_rents, save):
    """
    Create and save a plot showing the evolution of the proportion of happy agents.

    Params:
        happy_data(list or np.ndarray): Number of happy agents at each step.
        numagents(int): Total number of agents in the model.

    Returns: (None): The plot is saved to 'plots/happyness_evolution.png'.
    """

    output_path = "plots/rent_evolution.png"
    metrics_arr = np.array(district_rents)
    
    # Compute mean and variance across runs (axis=0) → shape (steps, 3)
    mean_vals = metrics_arr.mean(axis=0)
    var_vals  = metrics_arr.var(axis=0)
    
    steps = np.arange(mean_vals.shape[0])

    fig, ax = plt.subplots(figsize=(5, 3))

    # Dissimilarity (metric index 0)
    mean_district_0   = mean_vals[:, 0]
    sigma_district_0  = (np.sqrt(var_vals[:, 0])*1.96)/np.sqrt(len(metrics_arr))
    ax.plot(steps, mean_district_0, label="District 0", color = COLOURS[0])
    ax.fill_between(steps,
                    mean_district_0 - sigma_district_0,
                    mean_district_0 + sigma_district_0,
                    color = COLOURS[0],
                    alpha=0.2)
    
    # District 1
    mean_district_1    = mean_vals[:, 1]
    sigma_district_1   = (np.sqrt(var_vals[:, 1])*1.96)/np.sqrt(len(metrics_arr))
    ax.plot(steps, mean_district_1, label="District 1", color = COLOURS[1])
    ax.fill_between(steps,
                    mean_district_1 - sigma_district_1,
                    mean_district_1 + sigma_district_1,
                    color = COLOURS[1], 
                    alpha=0.2)
    
    # District 1
    mean_district_2    = mean_vals[:, 2]
    sigma_district_2  = (np.sqrt(var_vals[:, 2])*1.96)/np.sqrt(len(metrics_arr))
    ax.plot(steps, mean_district_2, label="District 2", color = COLOURS[2])
    ax.fill_between(steps,
                    mean_district_2 - sigma_district_2,
                    mean_district_2 + sigma_district_2,
                    color = COLOURS[2], 
                    alpha=0.2)
    
    
    ax.set_xlabel("Steps")
    ax.set_ylabel("Rent prices")
    ax.set_title("Evolution of Rent per district ")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(output_path, dpi=300)
    plt.close(fig)

def alpha_rents(last_metrics, alphas, save):

    output_path = "plots/alpha_rents.png"
    rent0_stats, rent1_stats, rent2_stats =  [], [], []
    fig, ax =  plt.subplots(figsize=(5, 3))
    for alpha, alpha_metric in last_metrics.items():

        rent0, rent1, rent2= zip(*alpha_metric)
        rent0_stats.append([np.mean(rent0), np.var(rent0)])
        rent1_stats.append([np.mean(rent1), np.var(rent1)])
        rent2_stats.append([np.mean(rent2), np.var(rent2)])
    
    rent0_stats = np.array(rent0_stats)
    rent1_stats = np.array(rent1_stats)
    rent2_stats = np.array(rent2_stats)

    names = ['A', 'B', 'C']

    for i, rent_stats in enumerate([rent0_stats, rent1_stats, rent2_stats]):
        ax.plot(alphas, rent_stats[:, 0], label=f"Rent of district {names[i]}", color = COLOURS[i])
        ax.fill_between(alphas,
                        rent_stats[:, 0] - (np.sqrt(rent_stats[:, 1])*1.96)/np.sqrt(len(alpha_metric)),
                        rent_stats[:, 0] + (np.sqrt(rent_stats[:, 1])*1.96)/np.sqrt(len(alpha_metric)),
                        alpha=0.2, color = COLOURS[i])
    
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Rent")
    ax.set_title("Rent of Ditricts After 2000 steps")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(output_path, dpi=300)
    plt.close(fig)

def alpha_metrics(last_metrics, alphas, save):
    output_path = "plots/alpha_evolution.png"
    exp_stat =  []
    diss_stat = []
    fig, ax =  plt.subplots(figsize=(5, 3))
    for _, alpha_metric in last_metrics.items():

        exposures, dissimilarities = zip(*alpha_metric)
        mean_exp = np.mean(exposures) 
        var_exp = np.var(exposures)
        mean_diss = np.mean(dissimilarities)
        var_diss = np.var(dissimilarities)
        exp_stat.append([mean_exp, var_exp])
        diss_stat.append([mean_diss, var_diss])
    
    exp_stat = np.array(exp_stat)
    diss_stat = np.array(diss_stat)
     # Exposure
    ax.plot(
        alphas,
        exp_stat[:, 0],
        label="Exposure",
        marker='o',       # ← add circles at each point
        linestyle='-',
        color =  COLOUR_MET[1]
    )
    ax.fill_between(
        alphas,
        exp_stat[:, 0] - (np.sqrt(exp_stat[:, 1])*1.96)/np.sqrt(len(alpha_metric)),
        exp_stat[:, 0] + (np.sqrt(exp_stat[:, 1])*1.96)/np.sqrt(len(alpha_metric)),
        alpha=0.2, 
        color = COLOUR_MET[1]
    )

    # Dissimilarity
    ax.plot(
        alphas,
        diss_stat[:, 0],
        label="Dissimilarity",
        marker='o',       # ← add circles here too
        linestyle='-', 
        color= COLOUR_MET[0]
    )
    ax.fill_between(
        alphas,
        diss_stat[:, 0] - (np.sqrt(diss_stat[:, 1])*1.96)/np.sqrt(len(alpha_metric)),
        diss_stat[:, 0] + (np.sqrt(diss_stat[:, 1])*1.96)/np.sqrt(len(alpha_metric)),
        alpha=0.2, 
        color= COLOUR_MET[0]
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Metric value")
    ax.set_title("Dissimilarity and Exposure After 2000 Steps")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save:
        fig.savefig(output_path, dpi=300)
    plt.close(fig)

def multiple_metrics_plot(stats_dict, save):
    
    fig, axs = plt.subplots(2, 2, figsize=(5.3,4), sharex=True, sharey=True)
    counter = 0
    axs = axs.flatten()
    for alpha, (stats_diss, stats_exp) in stats_dict.items():
        # if alpha%0.2 == 0 and alpha < 1: 
        mean_diss, sigma_diss = stats_diss
        steps = np.arange(len(mean_diss))
        axs[counter].plot(steps, mean_diss, label="Dissimilarity",  color=COLOUR_MET[0])
        axs[counter].fill_between(steps,
                    mean_diss - sigma_diss,
                    mean_diss + sigma_diss,
                    color=COLOUR_MET[0], 
                    alpha=0.2)
        mean_exp, sigma_exp = stats_exp
        axs[counter].plot(steps, mean_exp, label="Exposure", color=COLOUR_MET[1])
        axs[counter].fill_between(steps,
                    mean_exp - sigma_exp,
                    mean_exp + sigma_exp,
                     color=COLOUR_MET[1], 
                    alpha=0.2)
        axs[counter].set_title(r"$\alpha: $" + f"{alpha}")

        if counter > 1:
            axs[counter].set_xlabel("Step")

        if counter%2 == 0:
            axs[counter].set_ylabel("Metric value")
        
        if counter == 0:
            axs[counter].legend()
        counter +=1

        fig.suptitle("Evolution of Metrics")
        fig.tight_layout()
        if save:
            fig.savefig("plots/metrics_suplots.png", dpi=300)
        plt.close(fig)
        


def metrics_plot(metrics_diss_exp, alpha, save):
    """
    Create and save a plot showing the evolution of the proportion of happy agents.

    Params:
        happy_data(list or np.ndarray): Number of happy agents at each step.
        numagents(int): Total number of agents in the model.

    Returns: (None): The plot is saved to 'plots/happyness_evolution.png'.
    """

    output_path = f"plots/evolution_metrics/{alpha}.png"
    metrics_arr = np.array(metrics_diss_exp)
    
    # Compute mean and variance across runs (axis=0) → shape (steps, 2)
    mean_vals = metrics_arr.mean(axis=0)
    var_vals  = metrics_arr.var(axis=0)
    
    steps = np.arange(mean_vals.shape[0])
    fig, ax = plt.subplots(figsize=(5, 3))

    # Dissimilarity (metric index 0)
    mean_diss   = mean_vals[:, 1]
    sigma_diss  = (np.sqrt(var_vals[:, 1])*1.96)/np.sqrt(len(metrics_arr))
    ax.plot(steps, mean_diss, label="Dissimilarity",  color=COLOUR_MET[0])
    ax.fill_between(steps,
                    mean_diss - sigma_diss,
                    mean_diss + sigma_diss,
                    color=COLOUR_MET[0], 
                    alpha=0.2)
    
    # Exposure (metric index 1)
    mean_exp    = mean_vals[:, 0]
    sigma_exp   = (np.sqrt(var_vals[:, 0])*1.96)/np.sqrt(len(metrics_arr))
    ax.plot(steps, mean_exp, label="Exposure",  color=COLOUR_MET[1])
    ax.fill_between(steps,
                    mean_exp - sigma_exp,
                    mean_exp + sigma_exp,
                    color=COLOUR_MET[1], 
                    alpha=0.2)
    
    ax.set_xlabel("Steps")
    ax.set_ylabel("Metric value")
    ax.set_title(r"Dissimilarity and Exposure, $\alpha$: " + f"{alpha}")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return (mean_diss, sigma_diss), (mean_exp, sigma_exp)


def plot_income_distributions(base_incomes, sigma=0.25, n_samples=100, save=False):
    """
    Plot income distributions for given base incomes using a log-normal multiplier.

    Params:
        base_incomes (list of float): Base income values for each agent type.
        sigma (float): Standard deviation of the log-normal multiplier.
        n_samples (int): Number of samples to draw per base income.
    """
    plt.figure(figsize=(4,3))
    for i, base in enumerate(base_incomes):
        samples = base * np.random.lognormal(mean=0, sigma=sigma, size=n_samples)
        plt.hist(samples,
                 bins=30,
                 density=True,
                 alpha=0.5,
                 label=f'Base {base}', 
                color=COLOURS[i]
                 )
    plt.xlabel('Income')
    plt.ylabel('Density')
    plt.title('Income Distributions by Base Income')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig("plots/population_distribution.png", dpi=300)
    # plt.show()