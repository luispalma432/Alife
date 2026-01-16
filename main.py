import functions as fc

if __name__ == "__main__":
    #print('Starting Individual Evaluation...')
    #fc.individualAvaliation()
    #print('Finished Individual Evaluation...')
    print('Starting Genetic Algorithm Evaluation...')
    population_size = 100
    population = fc.generatePopulation(population_size)
    results = fc.geneticAlgorithm(population, generations=50, mutation_rate=0.01)
    print('Finished Genetic Algorithm Evaluation...')
    #Final population's strategy from each agent
    for i in results:
        print(i.strategy)
