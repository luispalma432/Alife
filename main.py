import networkx as nx
import orjson  # equivalente a json
import plotly
import polars as pl  # equivalente a pandas


def stratagies(value):
    """
     Aplica codigo de gray
     bit 3 e bit2 selecionar baseado na característica  nice
     bit 1 e bit 0 selecionar estrategia
    bit3 bit2 bit1 bit 0
     0    0    0    0     defect
     0    0    0    1     tester
     0    0    1    1     bully


    11 Nice:
        00 Compliant
        01 baseline
        11 tit or tat
        01 Compliant

    00 Nasty:
         0 0 Defect
         0 1 tester
         1 1 bully
         0 1 Defect

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

    # random, Scale Free, regularLattice, smallWorld
    # random- teste de estrategia de forma individual
    # Regular lattice rede 2D para cellular automata com deslocamento de moore
    # SmallWorld- teste de estrategias em comunidades
    #   variando o average path length e cluster coefficient
    # Scale Free  resultados em distribuições desiguais
    #  (presença de hubs e distribuições de população assimétrica e coesão preferencial)


def create_network(N, k, rewire_prob, scale_free=False, regularLattice=False):
    """
    N (int): Population Size (Must be perfect square for 2D Lattice).
    k (int): Average neighbors (Degree). Default 8 for Moore Neighborhood.
    rewire_prob (float): 0.0 to 1.0 (Controls Ring -> Small World -> Random).
    scale_free (bool): If True, overrides others to create Power Law hubs.
    2d_lattice (bool): If True, overrides SW/Random to create 2D Grid (Moore).
    """
    # case 1 Power Law
    if scale_free:
        m = int(k / 2)  # New edges per node
        return nx.barabasi_albert_graph(n=N, m=max(1, m), seed=42)

    # case 2 Regular lattice 2D

    # case 3 Watts-Strogatz Spectrum (Regular ring lattice, Small world, Random)

    pass


def genecticAlgorithm(population, mutation):
    """
    Verificar o desenvolvimento de populações e como afeta o sistema de Pontos
    Mutação: Mudaça de um dos parametros bit flip
              preferencial attacment
    """
    pass


"""
Random network is to evaluate best startagies
1X1

create a random network with all the stratagies

list stratagies codified
Aplica codigo de gray
bit 3 e bit2 selecionar baseado na característica  nice
bit 1 e bit 0 selecionar estrategia
bit3 bit2 bit1 bit 0
0    0    0    0     defect
0    0    0    1     tester
0    0    1    1     bully
0    0    1    0     Harrington


11 Nice:
   00 Compliant
   01 Tit for Two Tats 	TFTT
   11 tit or tat
   10 GRIM

00 Nasty:
    0 0 Defect
    0 1 tester
    1 1 bully
    1 0 Harrington

bit3 XOR bit2 random
"""

"""
Random network is to evaluate best startagies
1X1

create a random network with all the stratagies

list stratagies codified
Aplica codigo de gray
bit 3 e bit2 selecionar baseado na característica  nice
bit 1 e bit 0 selecionar estrategia
bit3 bit2 bit1 bit 0
0    0    0    0     defect
0    0    0    1     tester
0    0    1    1     bully


11 Nice:
   00 Compliant
   01 Tit for Two Tats 	TFTT
   11 tit or tat
   10 GRIM

00 Nasty:
    0 0 Defect
    0 1 tester
    1 1 bully
    1 0 Harrington

bit3 XOR bit2 random
"""
