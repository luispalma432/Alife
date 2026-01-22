import math
import random

import networkx as nx
import orjson  # equivalente a json
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
        return [1, 1]
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


def individualEvaluation():
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
        return nx.watts_strogatz_graph(n=N, k=k, p=1.0, seed=42)

    elif type == "SMALL_WORLD":
        # Note: 'side' isn't needed here, just return the graph
        return nx.watts_strogatz_graph(n=N, k=k, p=0.1, seed=42)

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


def populationEvaluation(population, type, population_List, node_Data_Map):
    iterations = 5
    nodes = population.number_of_nodes()

    # population = generatePopulations(N=nodes, k=neighbors, type=type)

    # population_List, node_Data_Map = generateEadgeList(population, stratagies_genomes)

    strategy_sum_scores = {}
    strategy_match_counts = {}
    strategy_node_counts = {}
    strategy_total_links = {}

    node_scores = {node_id: 0 for node_id in node_Data_Map}

    # Initialize Strategy Keys
    for g in stratagies_genomes:
        name = defineStrategies(g)
        if name not in strategy_sum_scores:
            strategy_sum_scores[name] = 0
            strategy_match_counts[name] = 0
            strategy_node_counts[name] = 0
            strategy_total_links[name] = 0

    for node_id, data in node_Data_Map.items():
        name = data["name"]
        degree = data["links"]
        strategy_node_counts[name] += 1
        strategy_total_links[name] += degree

    # This avoids calling Polars inside the loop (Massive Speedup)
    p1_names = population_List["p1_name"].to_list()
    p2_names = population_List["p2_name"].to_list()
    p1_ids = population_List["p1_id"].to_list()
    p2_ids = population_List["p2_id"].to_list()

    for _ in range(iterations):
        for p1_name, p2_name, p1_id, p2_id in zip(p1_names, p2_names, p1_ids, p2_ids):
            num_rounds = random.choice([180, 200, 220])
            scores = tournament(p1_name, p2_name, num_rounds)

            # Update Strategy Totals
            strategy_sum_scores[p1_name] += scores[0]
            strategy_sum_scores[p2_name] += scores[1]
            strategy_match_counts[p1_name] += 1
            strategy_match_counts[p2_name] += 1

            node_scores[p1_id] += scores[0]
            node_scores[p2_id] += scores[1]

    final_results = []

    for name in strategy_sum_scores:
        total_sum = strategy_sum_scores[name]
        matches = strategy_match_counts[name]
        node_count = strategy_node_counts[name]
        total_links = strategy_total_links[name]

        avg_score = total_sum / matches if matches > 0 else 0

        # Only include strategies that actually exist in this run
        if node_count > 0:
            final_results.append(
                {
                    "strategy": name,
                    "avg_tourney_score": round(avg_score, 2),
                    "node_count": node_count,
                    "total_links": total_links,
                    "unique_matches": total_links // 2,
                }
            )

    # Sort Descending (Best Strategy First)
    final_results.sort(key=lambda x: x["avg_tourney_score"], reverse=True)

    print(f"\n--- Population Evaluation ({type}) Population Size({nodes}) ---")
    print(f"{'Strategy':<15} | {'Avg Score':<12} | {'Nodes':<8} | {'Total Links':<12}")
    print("-" * 60)

    for res in final_results:
        print(
            f"{res['strategy']:<15} | {res['avg_tourney_score']:>12.1f} | {res['node_count']:>8} | {res['total_links']:>12}"
        )

    filename = f"population_{type.lower()}_results.json"
    with open(filename, "wb") as f:
        f.write(orjson.dumps(final_results, option=orjson.OPT_INDENT_2))

    return final_results, node_scores, population_List, node_Data_Map


def inicializeGa(N, K, type):
    inicial_population_graph = generatePopulations(N, K, type=type)

    # This creates the Gen 0 random strategies
    inicial_population_List, inicial_node_Data_Map = generateEadgeList(
        inicial_population_graph,
        stratagies_genomes,
    )

    finalResults, node_scores, population_List, node_Data_Map = populationEvaluation(
        population=inicial_population_graph,
        type=type,
        population_List=inicial_population_List,
        node_Data_Map=inicial_node_Data_Map,
    )

    # Return everything needed to start the GA Loop
    return (
        finalResults,
        node_scores,
        population_List,
        node_Data_Map,
        inicial_population_graph,
    )


def run_evolution_step(population_List, node_scores, current_map, mutation_rate=0.01):
    # Convert dictionary scores to DataFrame for fast joining
    scores_df = pl.DataFrame(
        {"node_id": list(node_scores.keys()), "score": list(node_scores.values())},
        schema={"node_id": pl.Int64, "score": pl.Float64},
    )  # Ensure types match your IDs

    df_a = population_List.select(
        [pl.col("p1_id").alias("node_id"), pl.col("p2_id").alias("neighbor_id")]
    )
    df_b = population_List.select(
        [pl.col("p2_id").alias("node_id"), pl.col("p1_id").alias("neighbor_id")]
    )
    neighbors_df = pl.concat([df_a, df_b])

    # Join Scores to see: [Me, My Score, Neighbor, Neighbor Score]
    neighbors_df = neighbors_df.join(
        scores_df.rename({"node_id": "neighbor_id", "score": "neighbor_score"}),
        on="neighbor_id",
        how="left",
    ).join(scores_df.rename({"score": "my_score"}), on="node_id", how="left")

    # Filter: Keep only neighbors who are strictly BETTER than me
    better_neighbors = neighbors_df.filter(
        pl.col("neighbor_score") > pl.col("my_score")
    )

    # Group By Node and pick the ONE neighbor with the Max Score
    best_imitation_df = (
        better_neighbors.sort("neighbor_score", descending=True)
        .unique(subset=["node_id"], keep="first")
        .select(["node_id", "neighbor_id"])
    )

    # Convert the "Winners List" to a dictionary for fast lookup: {Node_ID: Target_Neighbor_ID}
    # These are the nodes that decided to change their strategy
    updates_dict = dict(
        zip(
            best_imitation_df["node_id"].to_list(),
            best_imitation_df["neighbor_id"].to_list(),
        )
    )

    # CREATE NEXT GENERATION (Mutation & Update) ---
    new_node_map = {}

    for node_id, old_data in current_map.items():
        # Determine Base Genome (Copy or Keep)
        if node_id in updates_dict:
            target_id = updates_dict[node_id]
            new_genome = current_map[target_id]["genome"][:]
        else:
            new_genome = old_data["genome"][:]

        # Mutation (Bit Flip)
        if random.random() < mutation_rate:
            bit_to_flip = random.randint(0, 3)
            new_genome[bit_to_flip] = 1 - new_genome[bit_to_flip]

        # Save to New Map
        new_node_map[node_id] = {
            "genome": new_genome,
            "name": defineStrategies(new_genome),
            "links": old_data["links"],
        }

    return new_node_map


def update_edge_list(previous_df, current_node_map):
    """
    Updates the Polars DataFrame with the new strategies from the evolved map.
    It reuses the topology (p1_id, p2_id) from the previous generation for speed.
    """

    # We strip away the old strategy names/genomes, keeping only the IDs
    df = previous_df.select(["p1_id", "p2_id"])

    # We "paint" the new strategies onto the existing IDs
    df = df.with_columns(
        [
            # Update Player 1
            pl.col("p1_id")
            .map_elements(lambda x: current_node_map[x]["name"], return_dtype=pl.String)
            .alias("p1_name"),
            pl.col("p1_id")
            .map_elements(
                lambda x: current_node_map[x]["genome"], return_dtype=pl.List(pl.Int8)
            )
            .alias("p1_genome"),
            # Update Player 2
            pl.col("p2_id")
            .map_elements(lambda x: current_node_map[x]["name"], return_dtype=pl.String)
            .alias("p2_name"),
            pl.col("p2_id")
            .map_elements(
                lambda x: current_node_map[x]["genome"], return_dtype=pl.List(pl.Int8)
            )
            .alias("p2_genome"),
            # (Optional) Update Links count if you track it in the DF
            pl.col("p1_id")
            .map_elements(lambda x: current_node_map[x]["links"], return_dtype=pl.Int32)
            .alias("p1_links"),
            pl.col("p2_id")
            .map_elements(lambda x: current_node_map[x]["links"], return_dtype=pl.Int32)
            .alias("p2_links"),
        ]
    )

    return df


def gaSimulation(N, k, type, mutation_rate, generations):
    output = f"GAsimulation_results_{type}_{N}_{generations}.json"
    # structure to save final results from each generation in JSON
    data_results = []

    # incialize population first gen
    results, node_scores, population_List, current_map, inicial_population_graph = (
        inicializeGa(N=N, K=k, type=type)
    )
    data_results.append(results)

    for i in range(generations):
        print(f" GENERATION {i + 1}")
        next_gen_map = run_evolution_step(
            population_List, node_scores, current_map, mutation_rate=mutation_rate
        )

        population_List = update_edge_list(population_List, current_map)
        current_map = next_gen_map

        finalResults, node_scores, population_List, current_map = populationEvaluation(
            population=inicial_population_graph,
            type=type,
            population_List=population_List,
            node_Data_Map=current_map,
        )
        data_results.append(finalResults)

    try:
        with open(output, "wb") as f:
            f.write(orjson.dumps(data_results, option=orjson.OPT_INDENT_2))
        print(f"\nSUCCESS: Full simulation history saved to '{output}'")
    except Exception as e:
        print(f"ERROR: Could not save JSON file. {e}")

    return data_results


def run_pd_cellular_automaton_lattice(side, steps, init_coop_prob=0.5):
    """Run a Prisoner's Dilemma cellular automaton on a 2D lattice.

    Each node is either C or D (using the global C/D constants).
    At every time step:
      1. Each node plays PD against all 8 Moore neighbors (defined by the 2D_LATTICE graph).
      2. Payoffs are accumulated over all pairwise games.
      3. Synchronously, each node copies the strategy (C/D) of the
         highest‑payoff node in its neighborhood (including itself).

    Returns
    -------
    history : list[dict]
        Per‑step stats: fraction cooperators and average payoff.
    final_state : dict[int, int]
        Mapping node_id -> C or D at the last step.
    """

    # Build the lattice using your existing generator; k is ignored for 2D_LATTICE
    N = side * side
    G = generatePopulations(N=N, k=8, type="2D_LATTICE")

    # Random initial condition
    state = {
        node: (C if random.random() < init_coop_prob else D) for node in G.nodes()
    }

    history: list[dict] = []

    for t in range(steps):
        # 1) Compute payoffs by playing PD on every edge
        payoffs = {node: 0 for node in G.nodes()}
        for i, j in G.edges():
            p_i, p_j = scoring(state[i], state[j])
            payoffs[i] += p_i
            payoffs[j] += p_j

        # Record stats for this step
        num_coop = sum(1 for s in state.values() if s == C)
        avg_payoff = sum(payoffs.values()) / N
        history.append(
            {
                "step": t,
                "fraction_cooperators": num_coop / N,
                "avg_payoff": avg_payoff,
            }
        )

        # 2) Synchronous update: imitate best‑payoff neighbor (including self)
        new_state = {}
        for node in G.nodes():
            best_node = node
            best_payoff = payoffs[node]
            for neigh in G.neighbors(node):
                if payoffs[neigh] > best_payoff:
                    best_payoff = payoffs[neigh]
                    best_node = neigh
            new_state[node] = state[best_node]

        state = new_state

    return history, state
