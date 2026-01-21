import orjson

import functions as fc

"""
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


"""
data_results = fc.gaSimulation(
    N=1000, k=8, type="RANDOM", mutation_rate=0.01, generations=30
)
