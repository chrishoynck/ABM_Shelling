from mesa import Agent


class SchellingAgent(Agent):
    """Schelling segregation agent."""

    def __init__(
        self, unique_id, model, agent_type: int, homophily: float = 0.4, radius: int = 1
    ) -> None:
        """Create a new Schelling agent.
        Args:
            model: The model instance the agent belongs to
            agent_type: Indicator for the agent's type (minority=1, majority=0)
            homophily: Minimum number of similar neighbors needed for happiness
            radius: Search radius for checking neighbor similarity
        """
        super().__init__(unique_id, model)
        # self.cell = cell
        self.type = agent_type
        self.homophily = homophily
        self.radius = radius
        self.happy = False

    def assign_state(self) -> None:
        """Determine if agent is happy and move if necessary."""

        # neighbours = self.model.grid.get_neighbors(agent.pos, moore=True, include_center=False)
        neighbors = self.model.grid.get_neighbors(
            pos=self.pos,
            moore=True,
            include_center=False
        )
        # neighbors = list(self.cell.get_neighborhood(radius=self.radius).agents)

        # Count similar neighbors
        similar_neighbors = len([n for n in neighbors if n.type == self.type])

        # Calculate the fraction of similar neighbors
        if (valid_neighbors := len(neighbors)) > 0:
            similarity_fraction = similar_neighbors / valid_neighbors
        else:
            # If there are no neighbors, the similarity fraction is 0
            similarity_fraction = 0.0

        if similarity_fraction < self.homophily:
            self.happy = False
        else:
            self.happy = True
            self.model.happy += 1

    def step(self) -> None:
        # Move if unhappy
        if not self.happy:
            empty_cells = [
                (x, y)
                for cell_contents, x, y in self.model.grid.coord_iter()
                if len(cell_contents) == 0
            ]
            if empty_cells:
                new_position = self.random.choice(empty_cells)
                self.model.grid.move_agent(self, new_position)




# from mesa import Agent
# class SchellingAgent(Agent):
#     """Simple Schelling segregation agent."""
#     def __init__(self, unique_id, model, agent_type, homophily=0.4):
#         """
#         Simple Schelling segregation agent.

#         Params:
#             unique_id (int): Unique identifier for this agent.
#             model (mesa.Model): The model instance the agent belongs to.
#             agent_type (int): Indicator of the agent’s type (0, 1 or 2).
#             homophily (float): Minimum fraction of similar neighbours required for happiness.
#         """
#         super().__init__(unique_id, model)
#         self.type = agent_type
#         self.homophily = homophily
#         self.happy = False

#     def step(self):
#         """
#         Assess happiness and move to a random empty cell if unhappy.

#         At each step, the agent:
#           1. Gathers all neighbouring agents.
#           2. Computes the fraction of neighbours of the same type.
#           3. Sets self.happy based on whether that fraction ≥ self.homophily.
#           4. If unhappy, selects a random empty cell and moves there.
#         """
#         neighbor_nodes = self.model.grid.get_neighbors(self.pos)
#         neighbor_agents = []
#         for node_id in neighbor_nodes:
#             neighbor_agents.extend(
#                 self.model.grid.get_cell_list_contents([node_id])
#             )
#         if neighbor_agents:
#             similar = sum(1 for n in neighbor_agents if n.type == self.type)
#             frac = similar / len(neighbor_agents)
#         else:
#             frac = 0
#         self.happy = frac >= self.homophily
#         if not self.happy:
#             # look for empty cells
#             empties = [id for id in self.model.grid.G.nodes if self.model.grid.is_cell_empty(id)]
#             if empties:
#                 new_node = self.random.choice(empties)
#                 self.model.grid.move_agent(self, new_node)
