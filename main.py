import networkx as nx
import orjson  # equivalente a json
import plotly
import polars as pl  # equivalente a pandas


def stratagies(value):
    """
     Aplica codigo de gram
     bit 3 e bit2 selecionar baseado na carcteristica nice
     bit 1 e bit 0 selecionar estrategia
    bit3 bit2 bit1 bit 0
     0    0    0    0     defiant
     0    0    0    1     tester
     0    0    1    1     bully


    11 Nice:
        00 compliant
        01 baseline
        11 tit or tat

    00 Nasty:
         0 0 Defiant
         0 1 tester
         1 1 bully

    bit3 != bit2 random
    """


def tournoment(player1, player2, rounds):
    pass


def scoring():
    """
            Result(points)
    P1 P2  | P1 P2
    0  0   | 1  1
    0  1   | 0  5
    1  1   | 3  3
    1  0   | 0  5
    """
    pass


def population(random, powerLaw, regularLattice, smallWorld):
    # random, powerLaw, regularLattice, smallWorld
    # random- teste de estrategia de forma individual
    # Regular lattice rede 2D para cellular automata com deslocamento de moore
    # SmallWorld- teste de estrategias em comunidades
    #   variando o average path length e cluster coefficient
    # Power Law resultados em distribuições desiguais
    #  (presença de hubs e distribuições de população assimétrica e atacment preferencial)
    pass


def genecticAlgorithm(population, mutation, other):
    """
    Verificar o desenvolvimento de populações e como afeta o sistema de Pontos
    Mutação: Mudaça de um dos parametros bit flip
    """
    pass
