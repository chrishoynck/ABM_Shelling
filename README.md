# ABM_Schelling

An agent-based implementation of Schelling’s segregation model using a behavioural approach extended with  
spatial districts, dynamic housing prices, and happiness based on game-theoretic neighbourhood payoffs.  
Inspired by Utrecht’s district layout, agents balance social utility and rent bids to choose where to live, generating emergent segregation patterns.  
Rents are dynamically updated with a supply and demand mechanism 

## Key Features

- **Heterogeneous agents** with three income/types  
- **Neighbourhood game**: coordination vs. miscoordination payoffs (`pay_c`, `pay_m`)  
- **Dynamic rents**: agents submit willingness-to-pay bids; districts update prices based on its supply and the number of highest bids. 
- **Spatial districts**: custom regions (Utrecht outline)  
- **Global metrics**: dissimilarity index & exposure index over time  
- **Parallel & sensitivity analysis**: Sobol sampling via SALib and BatchRunner  

## Requirements
- Install via:
  ```bash
  pip install -r requirements.txt

## Repository Layout

```
.
├── data/                   GSA output 
├── src/
│   ├── agent.py            SchellingAgent logic & bidding  
│   ├── district.py         District rent & population management  
│   ├── model.py            SchellingModel core (grid, scheduling, metrics)  
│   └── visualization.py    Plots for happiness, rents, occupancy, metrics  
├── plots/                       
│   ├── animations/              GIF animations of grid evolutions  
│   │   └── snapshots/           Intermediate frames of animation
│   └── evolution_metrics/       Evolution of dissimilarity and exposure
├── Global_SAnb.ipynb       Global sensitivity analysis (Sobol indices)  
├── Schelling_3agents.py    Main driver: parameter sweep & parallel runs  
└── requirements.txt        Python dependencies
```

## Quickstart

1. **Run a single simulation**

   ```bash
   python Schelling_3agents.py
   ```

   Adjust parameters (grid size, `density`, `alpha`, payoffs, etc.) in `def main`.

2. **Visualize results**

   * Happiness over time
   * District price trajectories
   * Occupant distributions
   * Dissimilarity & exposure indices
     All via functions in `src/visualization.py`.

3. **Global Sensitivity Analysis**
   Launch `Global_SAnb.ipynb` to sample parameter space (e.g. `density`, `p_random`, `min_tenure`, `u_threshold`, `alpha`) and compute first/second/higher-order Sobol indices.

## Model Parameters

| Name                      | Description                                    |
| ------------------------- | ---------------------------------------------- |
| `width`, `height`         | Grid dimensions                                |
| `density`                 | Initial occupancy probability (0–1)            |
| `p_random`                | Chance of a random move when tenure expires    |
| `pay_c`, `pay_m`          | Payoffs for coordination vs. miscoordination   |
| `min_tenure`              | Minimum steps before reconsidering location    |
| `u_threshold`             | Happiness threshold for voluntary moves        |
| `alpha`                   | Weight on consumption (rent) vs. social payoff |
| `population_distribution` | Proportional mix of the three agent types      |
| `income_dist`             | Base income levels for each type               |

---
