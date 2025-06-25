import numpy as np
from src.model import SchellingModel
import src.visualization as vis
import random

from concurrent.futures import ProcessPoolExecutor
from functools import partial

# -------------------------- Simulation Setup ----------------------------- #
def run_model(which_run, steps, seedje, params):
    """
    Run the given model for a fixed number of steps, recording the grid state at each step.

    Params
    steps(int): Number of iterations to advance the model.
    model( mesa object):  Model instance providing a `.grid.coord_iter()` method and a `.step()` method.

    Returns: (List[np.ndarray]): A list of 2D arrays capturing the grid at each step, where
        empty cells are marked as –1 and occupied cells by agent type.
    """
    snapshots = []
    happyness = []

    run_seed = int(seedje+which_run)
    random.seed(run_seed)
    np.random.seed(run_seed)
    (width, height, density, p_random, pay_c, pay_m, max_tenure, u_threshold, alpha, pop_distribution, income_dist) = params
    model = SchellingModel(width, height, density, p_random, pay_c, pay_m, max_tenure, u_threshold, alpha, pop_distribution, income_dist, seedje = run_seed)
    run_metrics = []
    district_rents = []

    number_of_places_per_dist = [0,0,0]
    agent_type_per_district = [[0,0,0],[0,0,0],[0,0,0]]

    for s in range(steps):         
        # Capture grid state: -1 empty, 0, 1 or 2 agent types
        grid_state = np.full((width, height), -1)
        for cell in model.grid.coord_iter():
            content, x, y = cell
            if content:
                if s == steps-1:
                    number_of_places_per_dist[model.district_of[(x,y)].id] +=1
                    agent_type_per_district[model.district_of[(x,y)].id][content[0].type] +=1
                # capacity 1 grid, so take first agent
                grid_state[x, y] = content[0].type
        if s%10 == 0:
            snapshots.append(grid_state)
        
 
        district_rents.append((model.districts[0].rent,model.districts[1].rent, model.districts[2].rent ))
        happyness.append(model.happiness_per_type)
        run_metrics.append(model.metrics)
        model.step()

    print(f"happiness is reached after {s} steps.")
    # print(run_metrics)
    return snapshots, happyness, run_metrics, district_rents, model.num_agents_per_type, agent_type_per_district, number_of_places_per_dist, which_run


def execute_parallel_models(how_many, 
                            params,
                            seedje, 
                            steps):
    """
    Run multiple Schelling model simulations in parallel.

    Args:
        how_many (int): Number of runs to execute.
        params (dict): Model parameters.
        seedje (int): Base random seed.
        steps (int): Number of steps per run.

    Returns:
        list: Results from each parallel run.
    """

    print(f"starting parallel generation of Schelling model for {how_many} runs)")
    print("-----------------------------------------")
    runs = range(how_many)  # Create a range for the runs
    num_threads = min(how_many, 10)
    # Partially apply parameters for the worker function (currently one parameter setting, could vary with more)
    worker_function = partial(
        run_model,
        steps = steps,
        seedje = seedje,
        params = params
    )
        
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(worker_function, runs))
        
    return results


def unpack_and_parse_results(resultaatjes):
    """
    Align and pad runs to equal length, then compute total happiness per step.

    Params:
        resultaatjes (list of tuples):  
            Each element is (snapshots, happiness_measures, run_metrics,
            num_agents_grouped, run_id) for one run.

    Returns:
        tuple:
            snapshots_runs (Tuple): of snapshots per run;
            happiness_runs (List): padded happiness lists per run;
            run_metrics (List): padded metrics lists per run;
            num_agents_grouped (Tuple): agent counts per run.
    """

    # unpack results
    snapshots_runs, happiness_runs, run_metrics, district_rents, num_agents_grouped, agent_type_per_dist, num_plac_pdist, run_ids = zip(*resultaatjes)
    total_happy = []
    maximal_runs = max([len(x) for x in happiness_runs])

    # iterate through different runs
    for i, happiness_measures in enumerate(happiness_runs): 

        # if this is not the longest run, padd with last value
        if len(happiness_measures) < maximal_runs:
            diff = maximal_runs - len(happiness_measures)
            last_happy = happiness_measures[-1]
            last_metric = run_metrics[i][-1]
            last_rent = district_rents[i][-1]

            happiness_measures.extend([last_happy] * diff)
            run_metrics[i].extend([last_metric] * diff)
            district_rents[i].extend([last_rent] * diff)

        
        # compute total happiness (over all types)
        tot_happyness = np.sum(happiness_measures, axis=1)
        total_happy.append(tot_happyness)
    
    return snapshots_runs, happiness_runs, total_happy, run_metrics, district_rents, num_agents_grouped, np.array(agent_type_per_dist), num_plac_pdist
        


def main():


    # set parameters 
    width = 20
    height = 20
    density = 0.9

    # based on real data
    population_distribution = [0.1752,0.524,0.3008]
    # population_distribution = [0.09, 0.48, 0.43]
    income_dist =  [15.6, 41.2, 94.0]
    # vis.plot_income_distributions(income_dist, sigma=0.25, n_samples=1000)
    # income_dist = [5, 10, 20]
    
    p_random = 0.1
    pay_c = 10
    pay_m = 6
    max_tenure = 6
    u_threshold = 8
    # alphas = np.linspace(0, 1, 11)
    alphas = [0.0, 0.2, 0.5, 0.7]
    alpha = 0.5

    steps = 2000
    seedje = 43
    num_runs = 50
    params = (width, height, density, 
                p_random, pay_c, pay_m,
                max_tenure, u_threshold, alpha,
                population_distribution, income_dist)
    
    metrics_per_alpha = dict()
    rents_per_alpha = dict()
    stats_per_alpha = dict()
    # run singular model
    # run_model(0, steps, seedje, params)
    
    # parallel
    # collect all results over runs (parallelized implementation)
    for alpha in alphas:
        alpha = np.round(alpha,1)
        # pack params
        params = (width, height, density, 
                p_random, pay_c, pay_m,
                max_tenure, u_threshold, alpha,
                population_distribution, income_dist)
        
        if alpha>0: 
            print(f"RUNNING WITH DYNAMIC HOUSING PRICE: alpha = {alpha}")
        resultaatjes = execute_parallel_models(num_runs,
                                            params,
                                            seedje,
                                            steps)
        

        # collect data of run 
        snapshots_runs, happiness_runs, total_happy, run_metrics, district_rents,  num_agents_grouped, agent_type_per_dist, num_plc_per_dist = unpack_and_parse_results(resultaatjes)
        metrics_per_alpha[alpha] = [ metrics_list[-1] for metrics_list in run_metrics ]
        rents_per_alpha[alpha] =  [ rent_list[-1] for rent_list in district_rents ]
        # visualize

        vis.happyness_plot(total_happy, happiness_runs, num_agents_grouped)
        vis.district_prices_plot(district_rents)
        vis.occupants_per_dist(agent_type_per_dist, alpha, num_plc_per_dist)
        stats_diss, stats_exp = vis.metrics_plot(run_metrics, alpha)
        stats_per_alpha[alpha] = (stats_diss, stats_exp)
        vis.create_animation(snapshots_runs[0])
    
    # print(metrics_per_alpha)
    vis.multiple_metrics_plot(stats_per_alpha)
    vis.alpha_metrics(metrics_per_alpha, alphas)
    vis.alpha_rents(rents_per_alpha, alphas)
    

if __name__ == '__main__':
    main()

