from mesa import Agent
class SchellingAgent(Agent):
    """Simple Schelling segregation agent."""
    def __init__(self, unique_id, model, agent_type, homophily=0.4):
        """
        Simple Schelling segregation agent.

        Params:
            unique_id (int): Unique identifier for this agent.
            model (mesa.Model): The model instance the agent belongs to.
            agent_type (int): Indicator of the agent’s type (0 for majority, 1 for minority).
            homophily (float): Minimum fraction of similar neighbours required for happiness.
        """
        super().__init__(unique_id, model)
        self.type = agent_type
        self.homophily = homophily
        self.happy = False

    def step(self):
        """
        Assess happiness and move to a random empty cell if unhappy.

        At each step, the agent:
          1. Gathers all neighbouring agents.
          2. Computes the fraction of neighbours of the same type.
          3. Sets self.happy based on whether that fraction ≥ self.homophily.
          4. If unhappy, selects a random empty cell and moves there.
        """
        neighbor_nodes = self.model.grid.get_neighbors(self.pos)
        neighbor_agents = []
        for node_id in neighbor_nodes:
            neighbor_agents.extend(
                self.model.grid.get_cell_list_contents([node_id])
            )
        if neighbor_agents:
            similar = sum(1 for n in neighbor_agents if n.type == self.type)
            frac = similar / len(neighbor_agents)
        else:
            frac = 0
        self.happy = frac >= self.homophily
        if not self.happy:
            # look for empty cells
            empties = [id for id in self.model.grid.G.nodes if self.model.grid.is_cell_empty(id)]
            if empties:
                new_node = self.random.choice(empties)
                self.model.grid.move_agent(self, new_node)
