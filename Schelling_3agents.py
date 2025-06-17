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
        if model.happy < model.schedule.get_agent_count(): 
            model.step()
        else: 
            break 
    print(f"happiness is reached after {s} steps.")
    return snapshots, happyness, which_run


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
if __name__ == '__main__':
    main()
# # Simulation parameters
# width = 20
# height = 20
# density = 0.9
# income_distribution = [1/3, 1/3, 1/3]
# homophilies = [0.3, 0.3, 0.3]
# radius = 1
# steps = 100
# seedje = 43
# num_runs = 10 
# agent_counts = []

# # Initialize and run model

# params =  (width, height, density, income_distribution, homophilies, radius)
# resultaatjes = execute_parallel_models(num_runs, seedje, params)

# snapshots, happyness = run_model(steps, model)
# happy_people = np.sum(happyness, axis=1)

# --------------------------- Create Animation ----------------------------- #
# vis.create_animation(snapshots)
# vis.happyness_plot(happy_people, number_agents)


