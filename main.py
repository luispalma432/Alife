import random

import networkx as nx
import orjson  # equivalente a json
import plotly
import polars as pl  # equivalente a pandas

"""
     Result(points)
    P1 P2  | P1 P2
    0  0   | 1  1
    0  1   | 0  5
    1  1   | 3  3
    1  0   | 0  5
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


# Constants
C = 1
D = 0


def defineStrategies(genome):
    value = (genome[0] << 3) | (genome[1] << 2) | (genome[2] << 1) | genome[3]
    if 4 <= value <= 11:
        return "CHAOTIC"

    strategy_map = {
        0: "ALL_D",
        1: "TESTER",
        2: "HARRINGTON",
        3: "BULLY",
        12: "ALL_C",
        13: "TF2T",
        14: "GRIM",
        15: "TFT",
    }
    return strategy_map.get(value, "ERROR")


def getMove(stratagy, oppHistory, chaos_state):
    round_id = len(oppHistory)

    if stratagy == "ALL_D":
        return D
    elif stratagy == "ALL_C":
        return C
    elif stratagy == "CHAOTIC":
        return C if (chaos_state or 0.5) > 0.5 else D

    elif stratagy == "GRIM":
        return D if D in oppHistory else C

    elif stratagy == "TFT":
        return C if round_id == 0 else oppHistory[-1]

    elif stratagy == "TF2T":
        if round_id < 2:
            return C
        return D if (oppHistory[-1] == D and oppHistory[-2] == D) else C

    elif stratagy == "TESTER":
        if round_id == 0:
            return D
        return C if oppHistory[-1] == D else D

    elif stratagy == "HARRINGTON":
        return C if (round_id % 4 == 3) else D

    elif stratagy == "BULLY":
        if round_id == 0:
            return D
        return D if oppHistory[-1] == C else C

    return C


def scoring(move1, move2):
    if move1 == C and move2 == C:
        return [3, 3]
    if move1 == D and move2 == C:
        return [5, 0]
    if move1 == C and move2 == D:
        return [0, 5]
    if move1 == D and move2 == D:
        return [1, 1]  # Fixed typo: was 0 and 1
    return [0, 0]


def tournament(player1, player2, rounds):
    total_score1, total_score2 = 0, 0
    history1, history2 = [], []

    # Initialize chaos state
    chaos_state = random.random()

    for _ in range(rounds):  # Fixed: added range()
        p1_move = getMove(player1, history1, chaos_state)
        p2_move = getMove(player2, history2, chaos_state)

        res = scoring(p1_move, p2_move)
        total_score1 += res[0]
        total_score2 += res[1]  # Fixed: was res[0]

        history1.append(p2_move)
        history2.append(p1_move)

        # Update Chaos State (Logistic Map)
        chaos_state = 4.0 * chaos_state * (1.0 - chaos_state)

    return [total_score1, total_score2]


def individualAvaliation():
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

    strategy_sum_scores = {}
    iterations = 5

    # Initialize keys in dictionary
    for g in stratagies_genomes:
        name = defineStrategies(g)
        if name not in strategy_sum_scores:
            strategy_sum_scores[name] = 0

    for g1 in stratagies_genomes:
        for g2 in stratagies_genomes:
            p1 = defineStrategies(g1)
            p2 = defineStrategies(g2)

            for _ in range(iterations):
                num_rounds = random.choice([180, 200, 220])
                scores = tournament(p1, p2, num_rounds)

                strategy_sum_scores[p1] += scores[0]
                strategy_sum_scores[p2] += scores[1]

    # Analysis
    total_matches_per_strat = len(stratagies_genomes) * iterations
    final_results = []

    print(f"{'Strategy':<15} | {'Avg Tourney Score':<20}")
    print("-" * 40)

    for name, total_sum in strategy_sum_scores.items():
        avg_score = total_sum / total_matches_per_strat
        print(f"{name:<15} | {avg_score:>18.2f}")
        final_results.append({"strategy": name, "avg_tourney_score": avg_score})

    with open("individual_baseline.json", "wb") as f:
        f.write(orjson.dumps(final_results, option=orjson.OPT_INDENT_2))

    print("\nDONE: Results saved.")


if __name__ == "__main__":
    individualAvaliation()
