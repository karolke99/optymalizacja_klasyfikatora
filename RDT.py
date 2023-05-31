from sklearn.preprocessing import MinMaxScaler
import random
from util import get_training_results
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.tree import DecisionTreeClassifier
from util import get_training_results


def rdt_parameters(number_of_features, icls):
    genome = list()

    list_criterion = ['gini', 'entropy', 'log_loss']
    genome.append(list_criterion[random.randint(0, 2)])

    list_strategy = ["best", "random"]
    genome.append(list_strategy[random.randint(0, 1)])

    list_max_depth = random.randint(1, 600)
    genome.append(list_max_depth)

    list_minimum_samples_split = random.randint(2, 30)
    genome.append(list_minimum_samples_split)

    list_minimum_samples_leaf = random.randint(1, 30)
    genome.append(list_minimum_samples_leaf)

    list_max_features = ['sqrt', 'log2', random.uniform(0, 1), random.randint(0, 2000)]
    genome.append(list_max_features[random.randint(0, 3)])

    list_max_leaf_nodes = random.randint(2, 200)
    genome.append(list_max_leaf_nodes)

    list_min_impurity_decrease = random.uniform(1, 50)
    genome.append(list_min_impurity_decrease)

    return icls(genome)


def rdt_parameters_fitness(y, df, number_of_attributes, individual):
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    estimator = DecisionTreeClassifier(
        criterion=individual[0],
        splitter=individual[1],
        max_depth=individual[2],
        min_samples_split=individual[3],
        min_samples_leaf=individual[4],
        max_features=individual[5],
        max_leaf_nodes=individual[6],
        min_impurity_decrease=individual[7],
    )

    return get_training_results(df_norm, y, estimator)


def rdt_parameters_with_selection(number_of_features, icls):
    genome = list()

    list_criterion = ['gini', 'entropy', 'log_loss']
    genome.append(list_criterion[random.randint(0, 2)])

    list_strategy = ["best", "random"]
    genome.append(list_strategy[random.randint(0, 1)])

    list_max_depth = random.randint(1, 600)
    genome.append(list_max_depth)

    list_minimum_samples_split = random.randint(2, 30)
    genome.append(list_minimum_samples_split)

    list_minimum_samples_leaf = random.randint(1, 30)
    genome.append(list_minimum_samples_leaf)

    list_max_features = ['sqrt', 'log2', random.uniform(0, 1), random.randint(0, 2000)]
    genome.append(list_max_features[random.randint(0, 3)])

    list_max_leaf_nodes = random.randint(2, 200)
    genome.append(list_max_leaf_nodes)

    list_min_impurity_decrease = random.uniform(1, 50)
    genome.append(list_min_impurity_decrease)

    for i in range(0, number_of_features):
        genome.append(random.randint(0, 1))

    return icls(genome)


def mutation_rdt(individual):
    number_parameter = random.randint(0, len(individual) - 1)

    if number_parameter == 0:
        list_criterion = ['gini', 'entropy', 'log_loss']
        individual[0] = list_criterion[random.randint(0, 2)]
    elif number_parameter == 1:
        list_strategy = ["best", "random"]
        individual[1] = list_strategy[random.randint(0, 1)]
    elif number_parameter == 2:
        individual[2] = random.randint(1, 600)
    elif number_parameter == 3:
        individual[3] = random.randint(2, 30)
    elif number_parameter == 4:
        individual[4] = random.randint(1, 30)
    elif number_parameter == 5:
        list_max_features = ['sqrt', 'log2', random.uniform(0,1), random.randint(0,2000)]
        individual[5] = list_max_features[random.randint(0, 3)]
    elif number_parameter == 6:
        individual[6] = random.randint(2, 200)
    elif number_parameter == 7:
        individual[7] = random.uniform(1, 50)


def mutation_rdt_with_selection(individual):
    number_parameter = random.randint(0, len(individual) - 1)

    if number_parameter == 0:
        list_criterion = ['gini', 'entropy', 'log_loss']
        individual[0] = list_criterion[random.randint(0, 2)]
    elif number_parameter == 1:
        list_strategy = ["best", "random"]
        individual[1] = list_strategy[random.randint(0, 1)]
    elif number_parameter == 2:
        individual[2] = random.randint(1, 600)
    elif number_parameter == 3:
        individual[3] = random.randint(2, 30)
    elif number_parameter == 4:
        individual[4] = random.randint(1, 30)
    elif number_parameter == 5:
        list_max_features = ['sqrt', 'log2', random.uniform(0,1), random.randint(0,2000)]
        individual[5] = list_max_features[random.randint(0, 3)]
    elif number_parameter == 6:
        individual[6] = random.randint(2, 200)
    elif number_parameter == 7:
        individual[7] = random.uniform(0.01, 50)
    else:
        if individual[number_parameter] == 0:
            individual[number_parameter] = 1
        else:
            individual[number_parameter] = 0


def rdt_parameter_feature_fitness(y, df, number_of_attributes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)
    y = np.array(y)

    list_columns_to_drop = []

    for i in range(number_of_attributes, len(individual)):
        if individual[i] == 0:
            list_columns_to_drop.append(i - (len(individual) - number_of_attributes))
            # list_columns_to_drop.append(i - number_of_attributes)

    df_selected_features = df.drop(df.columns[list_columns_to_drop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df_selected_features)

    estimator = DecisionTreeClassifier(
        criterion=individual[0],
        splitter=individual[1],
        max_depth=individual[2],
        min_samples_split=individual[3],
        min_samples_leaf=individual[4],
        max_features=individual[5],
        max_leaf_nodes=individual[6],
        min_impurity_decrease=individual[7],
    )

    result_sum = 0

    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        cm = metrics.confusion_matrix(expected, predicted)
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        # tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()

        result = (tp + tn) / (tp + fp + tn + fn)
        result_sum = result_sum + result

    return result_sum / split,


class rdtManager:

    def get_parameters_with_selection(self):
        return rdt_parameters_with_selection

    def get_parameters(self):
        return rdt_parameters

    def get_parameters_fitness(self):
        return rdt_parameters_fitness

    def get_parameters_fitness_with_selection(self):
        return rdt_parameter_feature_fitness

    def get_mutation(self):
        return mutation_rdt

    def get_mutation_with_selection(self):
        return mutation_rdt_with_selection

    def get_default_classifier(self):
        return DecisionTreeClassifier()
