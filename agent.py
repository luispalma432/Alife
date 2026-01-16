import random

import functions as fc


class Agent:
    def __init__(self, genome=None):
        # If no genome is provided, generate a random 4-bit list
        self.genome = genome if genome else [random.randint(0, 1) for _ in range(4)]

        # Determine the strategy name based on the genome
        self.strategy = fc.defineStrategies(self.genome)

        # State variables for the simulation
        self.points = 0
        self.history = []
        self.chaos_state = random.random()  # Used if strategy is "CHAOTIC"

    def reset(self):
        """Resets history and points for a new round/tournament."""
        self.history = []
        self.points = 0