import networkx as nx
import orjson  # equivalente a json
import plotly
import polars as pl  # equivalente a pandas
import random

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

def defineStrategies(genome):
    # Convert list to integer
    # Logic: (b3*8) + (b2*4) + (b1*2) + (b0*1)
    value = (genome[0] << 3) | (genome[1] << 2) | (genome[2] << 1) | genome[3]

    # If the integer is between 4 and 11, bits 3 and 2 are different (01xx or 10xx)
    if 4 <= value <= 11:
        return "CHAOTIC"

    # Mapa em binário
    strategy_map = {
        # NASTY (00xx)
        0: "ALL_D",  # 0000
        1: "TESTER",  # 0001
        2: "HARRINGTON",  # 0010
        3: "BULLY",  # 0011
        # NICE (11xx)
        12: "ALL_C",  # 1100
        13: "TF2T",  # 1101
        14: "GRIM",  # 1110
        15: "TFT",  # 1111
    }

    return strategy_map.get(value, "ERROR")


def tournoment(player1, player2, rounds):
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


    op= random.choice([COOP, DEFECT])
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
   11 tit for tat
   10 GRIM

00 Nasty:
    0 0 Defect
    0 1 tester
    1 1 bully
    1 0 Harrington

bit3 XOR bit2 random
"""
import random


def getMove(stratagy, oppHistory, chaos_state):
    C = 1
    D = 0

    round_id = len(oppHistory)

    # --- NO HISTORY REQUIRED ---
    if stratagy == "ALL_D":
        return D

    elif stratagy == "ALL_C":
        return C

    elif stratagy == "CHAOTIC":
        if chaos_state is None:
            return random.choice([C, D])
        return C if chaos_state > 0.5 else D

    # --- HISTORY REQUIRED ---

    elif stratagy == "GRIM":
        if D in oppHistory:
            return D
        return C


    elif stratagy == "TFT":
        if round_id == 0:
            return C
        return oppHistory[-1]

    elif stratagy == "TF2T":
        if round_id < 2:
            return C
        #  Defect only if last 2 were D. Correct.
        if oppHistory[-1] == D and oppHistory[-2] == D:
            return D
        return C

    # --- NASTY STRATEGIES ---

    elif stratagy == "TESTER":
        if round_id == 0:
            return D
        #  If they fought back (D), apologize (C). Correct.
        if oppHistory[-1] == D:
            return C
        return D

    elif stratagy == "HARRINGTON":
        cycle_position = round_id % 4
        # Coop on 4th turn (index 3). Correct.
        if cycle_position == 3:
            return C
        return D

    elif stratagy == "BULLY":
        if round_id == 0:
            return D
        #  If they are nice, exploit them. Correct.
        if oppHistory[-1] == C:
            return D
        # If they fight back, back down.
        return C

    else:
        # Default safety
        return C







def defineStrategies(genome):
    # Convert list to integer
    # Logic: (b3*8) + (b2*4) + (b1*2) + (b0*1)
    value = (genome[0] << 3) | (genome[1] << 2) | (genome[2] << 1) | genome[3]

    # If the integer is between 4 and 11, bits 3 and 2 are different (01xx or 10xx)
    if 4 <= value <= 11:
        return "CHAOTIC"

    # Mapa em binário
    strategy_map = {
        # NASTY (00xx)
        0: "ALL_D",  # 0000
        1: "TESTER",  # 0001
        2: "HARRINGTON",  # 0010
        3: "BULLY",  # 0011
        # NICE (11xx)
        12: "ALL_C",  # 1100
        13: "TF2T",  # 1101
        14: "GRIM",  # 1110
        15: "TFT",  # 1111
    }

    return strategy_map.get(value, "ERROR")







def individualAvaliation(list):
    """
    Lista de tecnicas utilizadas realizar 1x1 e torneio com (180-220 rondas) 5X
    Media de pontos nos 5 torneios para cada estrategia

    [0000, 0001, 0011, 0010 , 1100, 1101, 1111, 1010, 0100, 1000]
    primeiros 2 bits seleciona
    """
    op=random.choice([COOP, DEFECT])

    pass
