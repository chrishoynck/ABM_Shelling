from mesa import Agent

class SchellingAgent(Agent):
    """Simple Schelling segregation agent."""
    def __init__(self, unique_id, model, agent_type, homophily=0.4, radius=1):
        super().__init__(unique_id, model)
        self.type = agent_type
        self.homophily = homophily
        self.radius = radius
        self.happy = False

    def step(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=self.radius
        )
        if neighbors:
            similar = sum(1 for n in neighbors if n.type == self.type)
            frac = similar / len(neighbors)
        else:
            frac = 0
        self.happy = frac >= self.homophily
        if self.happy: 
            self.model.happy += 1
            self.model.happiness_per_type[self.type] += 1
        if not self.happy:
            empties = list(self.model.grid.empties)
            if empties:
                new_pos = self.random.choice(empties)
                self.model.grid.move_agent(self, new_pos)
