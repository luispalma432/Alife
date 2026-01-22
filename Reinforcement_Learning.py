import numpy as np
import random

import functions as fc

#Test on an agent having a reinforcement learning model to play against a chaotic strategy agent

class Agent:
    def __init__(self, genome=None):
        # If no genome is provided, generate a random 4-bit list
        self.genome = genome if genome else [random.randint(0, 1) for _ in range(4)]

        # Determine the strategy name based on the genome
        self.strategy = fc.defineStrategies(self.genome)

        # State variables for the simulation
        self.points = 0
        self.history = []
        self.chaos_state = 4.0 * 0.8 * (1.0 - 0.8)  # Used if strategy is "CHAOTIC"

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
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.5):
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


def tournament(player1, player2, rounds):
    for _ in range(rounds):
        # 1. Get moves from both players
        p1_move = fc.getMove(player1.strategy, player2.history, player1.chaos_state)
        p2_move = fc.getMove(player2.strategy, player1.history, player2.chaos_state)

        # 2. Update histories
        player1.history.append(p1_move)
        player2.history.append(p2_move)

        res = fc.scoring(p1_move, p2_move)
        player1.points += res[0]
        player2.points += res[1]

        # Update Chaos State (Logistic Map)
        # chaos_state = 4.0 * chaos_state * (1.0 - chaos_state)
    return [player1.points, player2.points]

def tournament_RL(player1, rl_agent, rounds):
    reward = 0
    for _ in range(rounds):
        # 1. Get moves from both players
        p1_move = fc.getMove(player1.strategy, rl_agent.history, player1.chaos_state)
        pRL_move = rl_agent.choose_action(rl_agent.get_state(rl_agent.history, player1.history))

        print('Normal agent move:', p1_move, ' RL agent move:', pRL_move)

        # 2. Update histories
        player1.history.append(p1_move)
        rl_agent.history.append(pRL_move)

        # 3. Award points based on the scoring matrix
        if p1_move == 1 and pRL_move == 1:  # Both Cooperate
            player1.points += 8
            rl_agent.points += 8
            reward = 8
        elif p1_move == 0 and pRL_move == 0:  # Both Defect
            player1.points += 1
            rl_agent.points += 1
            reward = 1
        elif p1_move == 0 and pRL_move == 1:  # P1 Defects, P2 Cooperates
            player1.points += 5
            rl_agent.points += 0
            reward = -1
        elif p1_move == 1 and pRL_move == 0:  # P1 Cooperates, P2 Defects
            player1.points += 0
            rl_agent.points += 5
            reward = 5

        # Get the next state
        next_state = rl_agent.get_state(rl_agent.history, player1.history)
        # Uptade last state and action
        rl_agent.last_state = next_state
        rl_agent.last_action = rl_agent.history[-2] if not rl_agent.history else pRL_move
        # Uptade Q-Table
        rl_agent.learn(rl_agent.last_state, rl_agent.last_action, reward, next_state)

        # Update Chaos State (Logistic Map)
        player1.chaos_state = 4.0 * player1.chaos_state * (1.0 - player1.chaos_state)

    points = rl_agent.points - player1.points
    if points > 0:
        print(f"RL Agent won against {player1.strategy} by {points} points.")
    elif points < 0:
        print(f"{player1.strategy} won against RL Agent by {-points} points.")
    else:
        print(f"RL Agent tied against {player1.strategy}.")
    return [player1.points, rl_agent.points]

# Example usage
if __name__ == "__main__":
    # Create a chaotic strategy agent
    chaotic_agent = Agent(genome=[1, 0, 0, 0])  # CHAOTIC strategy

    # Create a reinforcement learning agent
    rl_agent = RLAgent(learning_rate=0.1, discount_factor=0.9, epsilon=0.2)

    # Run a tournament of 100 rounds
    tournament_RL(chaotic_agent, rl_agent, rounds=100)
    print("Final Q-Table:", rl_agent.q_table)
