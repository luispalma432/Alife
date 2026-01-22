import orjson

import functions as fc
import visualization
'''
data_results = fc.gaSimulation(
    N=1000, k=8, type="SMALL_WORLD", mutation_rate=0.01, generations=30
)

data_results = fc.gaSimulation(
    N=1000, k=8, type="2D_LATTICE", mutation_rate=0.01, generations=30
)
data_results = fc.gaSimulation(
    N=1000, k=8, type="SCALE_FREE", mutation_rate=0.01, generations=30
)
data_results = fc.gaSimulation(
    N=1000, k=8, type="RANDOM", mutation_rate=0.01, generations=30
)

data_results = fc.gaSimulation(
    N=1000, k=8, type="RANDOM", mutation_rate=0.01, generations=30
)
history, final_state = fc.run_pd_cellular_automaton_lattice(
    side=50,
    steps=200,
    init_coop_prob=0.8,
)

# CA in console
final_fraction_coop = history[-1]["fraction_cooperators"]
print("Final fraction of cooperators:", final_fraction_coop)
print("Average payoff at final step:", history[-1]["avg_payoff"])
'''
# CA Visualization
history = visualization.run_pd_ca_with_history(side=100, steps=50, init_coop_prob=0.9)
visualization.visualize_battle(history)
