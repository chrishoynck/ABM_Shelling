import numpy as np
from src.model import SchellingModel
import src.visualization as vis

from concurrent.futures import ProcessPoolExecutor
from functools import partial
PROCESSES = 10


# --------------------------- Simulation Setup ----------------------------- #
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
    (width, height, density, income_distribution, homophilies, radius) = params
    model = SchellingModel(width, height, density, income_distribution, homophilies, radius, seedje+which_run)
    run_metrics = []

    for s in range(steps):
        # Capture grid state: -1 empty, 0, 1 or 2 agent types
        grid_state = np.full((width, height), -1)
        for cell in model.grid.coord_iter():
            content, x, y = cell
            if content:
                # capacity 1 grid, so take first agent
                grid_state[x, y] = content[0].type
        snapshots.append(grid_state)
        happyness.append(model.happiness_per_type)
        run_metrics.append(model.metrics)
        if model.happy < model.schedule.get_agent_count(): 
            model.step()
        else: 
            break
    print(f"happiness is reached after {s} steps.")
    return snapshots, happyness, run_metrics, model.num_agents_per_type,  which_run


def execute_parallel_models(how_many, 
                            params,
                            seedje, 
                            steps):

    print(f"starting parallel generation of Schelling model for {how_many} runs)")
    print("-----------------------------------------")
    runs = np.arange(how_many)  # Create a range for the runs
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
    snapshots_runs, happiness_runs, run_metrics, num_agents_grouped, run_ids = zip(*resultaatjes)
    total_happy = []
    # total_agents = []
    maximal_runs = max([len(x) for x in happiness_runs])
    for i, happiness_measures in enumerate(happiness_runs): 
        if len(happiness_measures) < maximal_runs:
            diff = maximal_runs - len(happiness_measures)
            last_happy = happiness_measures[-1]
            last_metric = run_metrics[i][-1]
            happiness_measures.extend([last_happy] * diff)
            run_metrics[i].extend([last_metric] * diff)
            # print(len(run_metrics[i]))
        tot_happyness = np.sum(happiness_measures, axis=1)
        # tot_agents = np.sum(num_agents_grouped[i])
        total_happy.append(tot_happyness)
        # total_agents.append(tot_agents)
    
    return snapshots_runs, happiness_runs, total_happy, run_metrics, num_agents_grouped
        


def main():
    width = 20
    height = 20
    density = 0.9
    income_distribution = [1/3,1/3,1/3]

    homophilies = [0.3,0.3,0.3]
    radius = 1
    steps = 100
    seedje = 43
    num_runs = 10

    params = (width, height, density,
              income_distribution,
              homophilies, radius)
    resultaatjes = execute_parallel_models(num_runs,
                                           params,
                                           seedje,
                                           steps)
    # run_model(0, steps, seedje, params)

    # collect data of run 
    snapshots_runs, happiness_runs, total_happy, run_metrics,num_agents_grouped = unpack_and_parse_results(resultaatjes)
    

    vis.happyness_plot(total_happy, happiness_runs, num_agents_grouped)
    vis.metrics_plot(run_metrics)
    vis.create_animation(snapshots_runs[0])


if __name__ == '__main__':
    main()

