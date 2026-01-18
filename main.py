import math
import random

import networkx as nx
import orjson
import polars as pl
from networkx.classes.function import neighbors

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


if __name__ == "__main__":
    # fc.individualAvaliation()
    fc.populationAvaliation(nodes=100000, type="SCALE_FREE")
