import agent
import functions as fc

if __name__ == "__main__":
    #print('Starting Individual Evaluation...')
    #fc.individualAvaliation()
    #print('Finished Individual Evaluation...')
    '''
    print('Starting Genetic Algorithm Evaluation...')
    population_size = 100
    population = fc.generatePopulation(population_size)
    results = fc.geneticAlgorithm(population, generations=50, mutation_rate=0.01)
    print('Finished Genetic Algorithm Evaluation...')
    #Final population's strategy from each agent
    for i in results:
        print(i.strategy)
    strategy_definitions = {
        "ALL_D (Nasty)": [0, 0, 0, 0],
        "TESTER (Nasty)": [0, 0, 0, 1],
        "BULLY (Nasty)": [0, 0, 1, 1],
        "HARRINGTON (Nasty)": [0, 0, 1, 0],
        "ALL_C (Nice)": [1, 1, 0, 0],
        "TF2T (Nice)": [1, 1, 0, 1],
        "TFT (Nice)": [1, 1, 1, 1],
        "GRIM (Nice)": [1, 1, 1, 0],
        "CHAOTIC": [0, 1, 0, 0]
    }
    agents = []
    for name, genome in strategy_definitions.items():
        new_agent = agent.Agent(genome=genome)
        agents.append(new_agent)
    '''

    new_agent = agent.Agent(genome=[0, 1, 0, 0])  # CHAOTIC
    print('Starting Rl model Evaluation...')
    rl_agent = agent.RLAgent()
    fc.tournament_RL(new_agent, rl_agent, rounds=500)
    print('Q-Table:', rl_agent.q_table)
    print('Finished RL Model Evaluation...')