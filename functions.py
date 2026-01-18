import math
import random

import networkx as nx
import orjson  # equivalente a json
import plotly
import polars as pl  #

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

C = 1
D = 0


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


def tournament(player1, player2, rounds):
    total_score1, total_score2 = 0, 0
    history1, history2 = [], []

    # Initialize chaos state
    chaos_state = random.random()

    for _ in range(rounds):
        p1_move = getMove(player1, history1, chaos_state)
        p2_move = getMove(player2, history2, chaos_state)

        res = scoring(p1_move, p2_move)
        total_score1 += res[0]
        total_score2 += res[1]

        history1.append(p2_move)
        history2.append(p1_move)

        # Update Chaos State (Logistic Map)
        # chaos_state = 4.0 * chaos_state * (1.0 - chaos_state)

    return [total_score1, total_score2]


def individualAvaliation():
    strategy_sum_scores = {}
    strategy_match_counts = {}  # Added to track actual matches
    iterations = 5

    # Initialize dictionaries
    for g in stratagies_genomes:
        name = defineStrategies(g)
        if name not in strategy_sum_scores:
            strategy_sum_scores[name] = 0
            strategy_match_counts[name] = 0

    # Everyone vs Everyone Tournament
    for i, g1 in enumerate(stratagies_genomes):
        # Start from i to avoid playing the same match twice (A vs B and B vs A)
        # and to decide if you want strategies to play against themselves
        for j, g2 in enumerate(stratagies_genomes):
            p1 = defineStrategies(g1)
            p2 = defineStrategies(g2)

            for _ in range(iterations):
                num_rounds = random.choice([180, 200, 220])
                scores = tournament(p1, p2, num_rounds)

                strategy_sum_scores[p1] += scores[0]
                strategy_sum_scores[p2] += scores[1]

                strategy_match_counts[p1] += 1
                strategy_match_counts[p2] += 1

    final_results = []
    print("\n--- Corrected Individual Baseline ---")
    print(f"{'Strategy':<15} | {'Avg Tourney Score':<20}")
    print("-" * 40)

    for name in strategy_sum_scores:
        total_sum = strategy_sum_scores[name]
        total_matches = strategy_match_counts[name]

        # Calculate true average per tournament
        avg_score = total_sum / total_matches if total_matches > 0 else 0

        print(f"{name:<15} | {avg_score:>15.1f}")
        final_results.append({"strategy": name, "avg_tourney_score": avg_score})
    """
    # Save to JSON
    with open("individual_baseline_fixed.json", "wb") as f:
        f.write(orjson.dumps(final_results, option=orjson.OPT_INDENT_2))
    """


def generatePopulations(N, k, type):
    if type == "RANDOM":
        return nx.watts_strogatz_graph(n=N, k=k, p=1.0)

    elif type == "SMALL_WORLD":
        # Note: 'side' isn't needed here, just return the graph
        return nx.watts_strogatz_graph(n=N, k=k, p=0.1)

    elif type == "2D_LATTICE":
        side = int(math.sqrt(N))
        if side**2 != N:
            print(f"Adjusting N to {side * side} for a perfect square grid.")
        G = nx.grid_2d_graph(side, side, periodic=True)
        for x, y in list(G.nodes()):
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor = ((x + dx) % side, (y + dy) % side)
                G.add_edge((x, y), neighbor)
        return nx.convert_node_labels_to_integers(G)

    elif type == "SCALE_FREE":
        m = k // 2
        return nx.barabasi_albert_graph(n=N, m=m, seed=42)
    else:
        raise ValueError("Invalid option.")


def generateEadgeList(G, genomeList):
    node_data_map = {}
    for node in G.nodes():
        genome = random.choice(genomeList)
        node_data_map[node] = {
            "genome": genome,
            "name": defineStrategies(genome),
            "links": G.degree(node),
        }

    edges = list(G.edges())
    df = pl.DataFrame({"p1_id": [e[0] for e in edges], "p2_id": [e[1] for e in edges]})

    df = df.with_columns(
        [
            pl.col("p1_id")
            .map_elements(lambda x: node_data_map[x]["name"], return_dtype=pl.String)
            .alias("p1_name"),
            pl.col("p2_id")
            .map_elements(lambda x: node_data_map[x]["name"], return_dtype=pl.String)
            .alias("p2_name"),
            pl.col("p1_id")
            .map_elements(
                lambda x: node_data_map[x]["genome"], return_dtype=pl.List(pl.Int8)
            )
            .alias("p1_genome"),
            pl.col("p2_id")
            .map_elements(
                lambda x: node_data_map[x]["genome"], return_dtype=pl.List(pl.Int8)
            )
            .alias("p2_genome"),
            pl.col("p1_id")
            .map_elements(lambda x: node_data_map[x]["links"], return_dtype=pl.Int32)
            .alias("p1_links"),  # Added
            pl.col("p2_id")
            .map_elements(lambda x: node_data_map[x]["links"], return_dtype=pl.Int32)
            .alias("p2_links"),
        ]
    )

    return df, node_data_map


def populationAvaliation(type, nodes):
    strategy_sum_scores = {}
    strategy_match_counts = {}  # Total times strategy played (Stubs)
    strategy_node_counts = {}  # NEW: Number of players with this strategy
    strategy_total_links = {}  # Sum of degrees (Total Stubs)

    iterations = 5
    neighbors = 8

    # Setup Network
    population = generatePopulations(N=nodes, k=neighbors, type=type)
    population_List, node_Data_Map = generateEadgeList(population, stratagies_genomes)

    # Initialize Dictionaries
    for g in stratagies_genomes:
        name = defineStrategies(g)
        if name not in strategy_sum_scores:
            strategy_sum_scores[name] = 0
            strategy_match_counts[name] = 0
            strategy_node_counts[name] = 0
            strategy_total_links[name] = 0

    #  Aggregation Loop (The "Occurrences" Logic)
    # We iterate through the MAP, not the edge list, to count nodes and degrees.
    for node_id, data in node_Data_Map.items():
        name = data["name"]
        degree = data["links"]

        # Count the Agent
        strategy_node_counts[name] += 1
        # Sum the Degrees (The "Social Reach")
        strategy_total_links[name] += degree

    # Tournament Loop
    for _ in range(iterations):
        for row in population_List.iter_rows(named=True):
            p1_name = row["p1_name"]
            p2_name = row["p2_name"]
            num_rounds = random.choice([180, 200, 220])

            scores = tournament(p1_name, p2_name, num_rounds)

            strategy_sum_scores[p1_name] += scores[0]
            strategy_sum_scores[p2_name] += scores[1]
            strategy_match_counts[p1_name] += 1
            strategy_match_counts[p2_name] += 1

    #  Analysis & Print
    final_results = []
    print(f"\n--- Population Evaluation ({type}) Population Size({nodes}) ---")
    print(f"{'Strategy':<15} | {'Avg Score':<12} | {'Nodes':<8} | {'Total Links':<12}")
    print("-" * 60)

    for name in strategy_sum_scores:
        total_sum = strategy_sum_scores[name]
        matches = strategy_match_counts[name]

        # Data from our aggregation loop
        node_count = strategy_node_counts[name]
        total_links = strategy_total_links[name]

        # Calculate Average Score per Match
        avg_score = total_sum / matches if matches > 0 else 0

        print(f"{name:<15} | {avg_score:>12.1f} | {node_count:>8} | {total_links:>12}")

        final_results.append(
            {
                "strategy": name,
                "avg_tourney_score": round(avg_score, 2),
                "node_count": node_count,
                "total_links": total_links,  # This is the degree sum (Stubs)
                "unique_matches": total_links // 2,  # Optional: Physical edges
            }
        )
    """
    # Save Results
    filename = f"population_{type.lower()}_results.json"
    with open(filename, "wb") as f:
        f.write(orjson.dumps(final_results, option=orjson.OPT_INDENT_2))

    print(f"\nDONE: Results saved to {filename}\n")
    """
