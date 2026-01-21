import numpy as np
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

# Reinforcement Learning Agent
# ACTION SPACE = {1 = Cooperate, 0 = Defect}
# STATE SPACE = {CC, CD, DC, DD}
# REWARD = {3,0,5,1}
#STEP-BY-STEP:
#1. Define transition matrix
#2. Initialize Q-table
#3. For each round:
#   a. Observe current state
#   b. Choose action based on epsilon-greedy policy
#   c. Execute action and observe reward and next state
#   d. Update Q-value using the Q-learning formula

class RLAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.points = 0
        self.history = []
        self.strategy = "RL_LEARNER"

        # Hyperparameters
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon  # Exploration rate (chance to try random move)

        # State mapping: {CC: 0, CD: 1, DC: 2, DD: 3}
        # Action mapping: {Defect: 0, Cooperate: 1}
        self.q_table = np.zeros((4, 2))
        self.last_state = None
        self.last_action = None

    def get_state(self, my_history, opp_history):
        if not my_history or not opp_history:
            return None  # No history yet

        # Look at the very last round to determine current state
        me = my_history[-1]
        opp = opp_history[-1]

        if me == 1 and opp == 1: return 0  # CC
        if me == 1 and opp == 0: return 1  # CD
        if me == 0 and opp == 1: return 2  # DC
        if me == 0 and opp == 0: return 3  # DD
        return None

    def choose_action(self, state):
        # Epsilon-greedy strategy: explore or exploit
        if state is None or random.random() < self.epsilon:
            return random.randint(0, 1)  # Random move
        else:
            return np.argmax(self.q_table[state])  # Best known move

    def learn(self, state, action, reward, next_state):
        if state is None: return

        # Predict the "future" value of the next state
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]

        # Update the Q-value for the action we just took
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])

    def reset(self):
        self.history = []
        self.points = 0