import multiprocessing

import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import random

from deap import base
from deap import creator
from deap import tools

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

df = pd.read_csv("data.csv", sep=",")
y = df['Status']

df.drop('Status', axis=1, inplace=True)
df.drop('ID', axis=1, inplace=True)
df.drop('Recording', axis=1, inplace=True)

number_of_attributes = len(df.columns)
#
mms = MinMaxScaler()
df_norm = mms.fit_transform(df)


#
# clf = SVC()
# scores = model_selection.cross_val_score(clf, df_norm, y, cv=5, scoring='accuracy', n_jobs=-1)
#
# print(scores.mean())

def generate_mean_plot(mean_list):
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(range(len(mean_list)), mean_list)
    plt.xlabel("epoch")
    plt.ylabel("mean")
    fig.savefig('./mean.png')  # save the figure to file
    plt.close(fig)


def generate_standard_deviation_plot(standard_deviation_list):
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(range(len(standard_deviation_list)), standard_deviation_list)
    plt.xlabel("epoch")
    plt.ylabel("standard deviation")
    fig.savefig('./standard_deviation.png')  # save the figure to file
    plt.close(fig)


def generate_best_value_plot(best_value):
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(range(len(best_value)), best_value)
    plt.xlabel("epoch")
    plt.ylabel("best value")
    fig.savefig('./best_value.png')  # save the figure to file
    plt.close(fig)



def SVCParametersFeatures(number_of_features, icls):
    genome = list()

    # kernel
    list_kernel = ["linear", "rbf", "poly", "sigmoid"]
    genome.append(list_kernel[random.randint(0, 3)])

    # c
    k = random.uniform(0.1, 100)
    genome.append(k)

    # degree
    genome.append((random.randint(1, 5)))

    # gamma
    gamma = random.uniform(0.001, 5)
    genome.append(gamma)

    #coeff
    coeff = random.uniform(0.01, 10)
    genome.append(coeff)

    for i in range(0, number_of_features):
        genome.append(random.randint(0, 1))

    return icls(genome)


def SVCParameterFeatureFitness(y, df, number_of_attributes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    list_columns_to_drop = []

    for i in range(number_of_attributes, len(individual)):
        if individual[i] == 0:
            list_columns_to_drop.append(i-number_of_attributes)

    df_selected_features = df.drop(df.columns[list_columns_to_drop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    estimator = SVC(
        kernel=individual[0],
        C=individual[1],
        degree=individual[2],
        gamma=individual[3],
        coef0=individual[4],
        random_state=101
    )

    result_sum = 0

    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()

        result = (tp + tn) / (tp + fp + tn + fn)
        result_sum = result_sum + result

    return result_sum / split,


def mutationSVC(individual):
    number_parameter = random.randint(0, len(individual) - 1)

    if number_parameter == 0:
        # kernel
        list_kernel = ["linear", "rbf", "poly", "sigmoid"]
        individual[0] = list_kernel[random.randint(0, 3)]
    elif number_parameter == 1:
        # C
        k = random.uniform(0.1, 100)
        individual[1] = k
    elif number_parameter == 2:
        # degree
        individual[2] = random.randint(1, 5)
    elif number_parameter == 3:
        # gamma
        gamma = random.uniform(0.01, 5)
        individual[3] = gamma
    elif number_parameter == 4:
        # coeff
        coeff = random.uniform(0.1, 20)
        individual[2] = coeff
    else:
        if individual[number_parameter] == 0:
            individual[number_parameter] = 1
        else:
            individual[number_parameter] = 0


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register('individual', SVCParametersFeatures, number_of_attributes, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", SVCParameterFeatureFitness, y, df, number_of_attributes)

# selTournament(tournsize = ), selRandom(k = ), selBest(k = ), selWorst(k = ), selRoulette(k = )
toolbox.register("select", tools.selTournament, tournsize=5)

toolbox.register("mutate", mutationSVC)

sizePopulation = 100
probabilityMutation = 0.4
probabilityCrossover = 0.8
numberIteration = 100

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
            # mutate an individual with probability MUTPB
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