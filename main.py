import functions as fc


def generateFiles():
    # --- BASELINE SCENARIOS (Standard Control Group) ---
    # Standard Small World (Social Network Analogy)
    # fc.gaSimulation(N=1000, k=8, type="SMALL_WORLD", mutation_rate=0.01, generations=30)
    # Standard Random (No structure)
    # fc.gaSimulation(N=1000, k=8, type="RANDOM", mutation_rate=0.01, generations=30)
    # Standard Scale-Free (Influencer/Hub dynamic)
    # fc.gaSimulation(N=1000, k=8, type="SCALE_FREE", mutation_rate=0.01, generations=30)

    #  CONNECTIVITY (Sparse vs Dense) ---
    # Sparse Society: Harder for cooperation to spread, easier for hubs to dominate
    """
    fc.gaSimulation(
        N=100000, k=4, type="SCALE_FREE", mutation_rate=0.01, generations=1000
    )
    """
    # Dense Society: High connectivity (k=20) typically helps Cooperators (Clusters form easier)
    # fc.gaSimulation( N=1000, k=20, type="SMALL_WORLD", mutation_rate=0.01, generations=40)

    #  MUTATION CHAOS (Stability vs Noise) ---
    # High Mutation (5%): Strategies will be unstable; "Nice" strategies might struggle to hold territory.
    # fc.gaSimulation(N=1000, k=8, type="RANDOM", mutation_rate=0.05, generations=50)
    # Ultra-Low Mutation (0.1%): Once a strategy wins, it stays won. Very stable.
    # fc.gaSimulation(N=1000, k=8, type="SMALL_WORLD", mutation_rate=0.001, generations=50)

    # SPATIAL DYNAMICS (The "Petri Dish") ---
    # Large 2D Grid (50x50 = 2500 nodes):
    # Ideal for seeing "Waves" of cooperation chasing defection (Cellular Automata effect)
    # fc.gaSimulation(N=2500, k=8, type="2D_LATTICE", mutation_rate=0.01, generations=60)

    # THE "VILLAGE" (Small & Tight) ---
    # Small population, high generations. Can a single "Bully" take over the whole village?
    # fc.gaSimulation(N=200, k=4, type="SMALL_WORLD", mutation_rate=0.02, generations=100)
    # fc.gaSimulation(N=1000, k=4, type="RANDOM", mutation_rate=0.02, generations=100)
    fc.gaSimulation(N=100, k=3, type="RANDOM", mutation_rate=0.02, generations=30)


generateFiles()


def Ca():
    """Inicalizar a população
    Gerar população create eadge list já tenho
    regras para o CA função não tenho
    update eadge list
    """
