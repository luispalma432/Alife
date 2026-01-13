import random
from agent import Agent
from main import tournoment


def run_ga_generation(population, generations=50, mutation_rate=0.01):
    for gen in range(generations):
        # 1. Battle Phase (Round Robin for Stage 1)
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                tournoment(population[i], population[j], rounds=200)

        # 2. Selection: Sort by points and keep top 50%
        population.sort(key=lambda a: a.points, reverse=True)
        next_gen_parents = population[:50]

        print(f"Gen {gen+1} Top Strategy: {next_gen_parents[0].strategy} ({next_gen_parents[0].points} pts)")

        # 3. Reproduction
        new_population = []
        while len(new_population) < 100:
            p1, p2 = random.sample(next_gen_parents, 2)

            # Crossover (Bit Splitting)
            point = random.randint(1, 3)
            child_genome = p1.genome[:point] + p2.genome[point:]

            # Mutation (Bit Flip)
            if random.random() < mutation_rate:
                idx = random.randint(0, 3)
                child_genome[idx] = 1 - child_genome[idx]

            new_population.append(Agent(genome=child_genome))

        population = new_population
    return population


# Initial Setup
pop = [Agent() for _ in range(100)]
final_pop = run_ga_generation(pop)
