import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import functions as fc
from matplotlib.colors import ListedColormap

def run_pd_ca_with_history(side, steps, init_coop_prob=0.5):
    N = side * side
    # Using your existing graph generator
    G = fc.generatePopulations(N=N, k=8, type="2D_LATTICE")

    # Initialize state: C=1 (Blue), D=0 (Red)
    state_map = {node: (1 if random.random() < init_coop_prob else 0) for node in G.nodes()}

    grid_history = []

    for t in range(steps):
        # Convert dict to a 2D numpy array for easy plotting
        current_grid = np.zeros((side, side))
        for (r, c), node in zip(np.ndindex(side, side), G.nodes()):
            current_grid[r, c] = state_map[node]
        grid_history.append(current_grid)

        # 1) Scoring phase
        payoffs = {node: 0 for node in G.nodes()}
        for i, j in G.edges():
            # scoring() returns [p1, p2]
            p_i, p_j = fc.scoring(state_map[i], state_map[j])
            payoffs[i] += p_i
            payoffs[j] += p_j

        # 2) Imitation phase (The "Fight")
        new_state = {}
        for node in G.nodes():
            best_node = node
            max_payoff = payoffs[node]
            for neighbor in G.neighbors(node):
                if payoffs[neighbor] > max_payoff:
                    max_payoff = payoffs[neighbor]
                    best_node = neighbor
            new_state[node] = state_map[best_node]
        state_map = new_state

    return grid_history


def visualize_battle(grid_history):
    fig, ax = plt.subplots(figsize=(8, 8))

    # 0 = Red (Defect/Nasty), 1 = Blue (Cooperate/Nice)
    # Based on your scoring: T=5, R=3, P=1, S=0
    cmap = ListedColormap(['#6b2323', '#1f2382'])

    im = ax.imshow(grid_history[0], cmap=cmap, interpolation='nearest')
    ax.axis('off')

    def update(frame):
        # Update the grid data
        im.set_data(grid_history[frame])

        # Calculate statistics
        num_blue = int(np.sum(grid_history[frame]))
        total = grid_history[frame].size
        percent = (num_blue / total) * 100

        # We set the title on the axis directly.
        # Removing blit=True makes this update reliably.
        ax.set_title(f"Step {frame} | Blue (C): {num_blue} ({percent:.1f}%)", fontsize=14)

        return [im]

    # interval=2000 (2 seconds) as you requested
    # blit=False ensures the title and text always update correctly
    ani = animation.FuncAnimation(fig, update, frames=len(grid_history),
                                  interval=2000, blit=False)
    plt.show()