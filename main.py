import orjson

import functions as fc

stratagies_genomes = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 0],  # Nasty
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 0],  # Nice
    [0, 1, 0, 0],
    [1, 0, 0, 0],  # Chaotic
]

"""
função
numero de nós
average degree
tipo de população
mutation rate
iterations

"""


data_results = fc.gaSimulation(
    N=1000, k=8, type="SCALE_FREE", mutation_rate=0.01, generations=30
)
