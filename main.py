import random
import warnings

from deap import base
from deap import creator
from deap import tools
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from KNN import KnnManager
from MLP import MlpManager
from util import generate_mean_plot, generate_standard_deviation_plot, generate_best_value_plot
import multiprocessing

pd.set_option('display.max_columns', None)

df = pd.read_csv("data.csv", sep=",")
y = df['Status']

df.drop('Status', axis=1, inplace=True)
df.drop('ID', axis=1, inplace=True)
df.drop('Recording', axis=1, inplace=True)

number_of_attributes = len(df.columns)

# configuration here
sizePopulation = 100
probabilityMutation = 0.4
probabilityCrossover = 0.8
numberIteration = 50
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
manager = MlpManager()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('individual', manager.get_parameters(), number_of_attributes, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", manager.get_parameters_fitness(), y, df, number_of_attributes)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mutate", manager.get_mutation())

pop = toolbox.population(n=sizePopulation)

fitness = list(toolbox.map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitness):
    ind.fitness.values = fit

g = 0
numberElitism = 1

mean_record = list()
std_record = list()
min_record = list()

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)
    toolbox.register("map", pool.map)

    while g < numberIteration:
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(toolbox.map(toolbox.clone, offspring))

        listElitism = []
        for x in range(0, numberElitism):
            listElitism.append(tools.selBest(pop, 1)[0])

        for mutant in offspring:
            if random.random() < probabilityMutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        print(" Evaluated %i individuals" % len(invalid_ind))
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        min_record.append(max(fits))
        mean_record.append(mean)
        std_record.append(std)

        print(" Min %s" % min(fits))
        print(" Max %s" % max(fits))
        print(" Avg %s" % mean)
        print(" Std %s" % std)

        best_ind = tools.selBest(pop, 1)[0]

        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        print("-- End of (successful) evolution --")

    generate_mean_plot(mean_record)
    generate_standard_deviation_plot(std_record)
    generate_best_value_plot(min_record)