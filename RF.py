from sklearn.preprocessing import MinMaxScaler
import random
from util import get_training_results
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.ensemble import RandomForestClassifier
from util import get_training_results


def rf_parameters(number_of_features, icls):
    genome = list()

    list_of_trees = random.randint(1, 5)
    genome.append(list_of_trees)

    list_criterion = ['gini', 'entropy', 'log_loss']
    genome.append(list_criterion[random.randint(0, 2)])

    list_max_depth = random.randint(1, 600)
    genome.append(list_max_depth)

    list_minimum_samples_split = random.randint(2, 30)
    genome.append(list_minimum_samples_split)

    list_minimum_samples_leaf = random.randint(1, 30)
    genome.append(list_minimum_samples_leaf)

    list_minimum_fraction_leaf = random.uniform(0.1, 0.5)
    genome.append(list_minimum_fraction_leaf)

    list_max_features = ['sqrt', 'log2', random.uniform(0.1, 1), random.randint(1, number_of_features)]
    genome.append(list_max_features[random.randint(0, 3)])

    list_max_leaf_nodes = random.randint(2, 200)
    genome.append(list_max_leaf_nodes)

    list_min_impurity_decrease = random.uniform(1, 50)
    genome.append(list_min_impurity_decrease)

    list_bootstrap = [True, False]
    genome.append(list_bootstrap[random.randint(0, 1)])

    if genome[-1]:
        list_oob_score = [True, False]
        genome.append(list_oob_score[random.randint(0, 1)])
    else:
        genome.append(0)

    list_verbose = random.randint(1, 50)
    genome.append(list_verbose)

    list_warm_start = [True, False]
    genome.append(list_warm_start[random.randint(0, 1)])

    list_ccp_alpha = random.uniform(1, 10)
    genome.append(list_ccp_alpha)

    return icls(genome)


def rf_parameters_fitness(y, df, number_of_attributes, individual):
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    estimator = RandomForestClassifier(
        n_estimators = individual[0],
        criterion = individual[1],
        max_depth = individual[2],
        min_samples_split = individual[3],
        min_samples_leaf = individual[4],
        min_weight_fraction_leaf = individual[5],
        max_features = individual[6],
        max_leaf_nodes = individual[7],
        min_impurity_decrease = individual[8],
        bootstrap = individual[9],
        # oob_score = individual[10],
        verbose = individual[11],
        warm_start = individual[12],
        ccp_alpha = individual[13],
    )

    return get_training_results(df_norm, y, estimator)


def rf_parameters_with_selection(number_of_features, icls):
    genome = list()

    list_of_trees = random.randint(1, 5)
    genome.append(list_of_trees)

    list_criterion = ['gini', 'entropy', 'log_loss']
    genome.append(list_criterion[random.randint(0, 2)])

    list_max_depth = random.randint(1, 600)
    genome.append(list_max_depth)

    list_minimum_samples_split = random.randint(2, 30)
    genome.append(list_minimum_samples_split)

    list_minimum_samples_leaf = random.randint(1, 30)
    genome.append(list_minimum_samples_leaf)

    list_minimum_fraction_leaf = random.uniform(0.1, 0.5)
    genome.append(list_minimum_fraction_leaf)

    list_max_features = ['sqrt', 'log2', random.uniform(0.1, 1), random.randint(1, number_of_features)]
    genome.append(list_max_features[random.randint(0, 3)])

    list_max_leaf_nodes = random.randint(2, 200)
    genome.append(list_max_leaf_nodes)

    list_min_impurity_decrease = random.uniform(1, 50)
    genome.append(list_min_impurity_decrease)

    list_bootstrap = [True, False]
    genome.append(list_bootstrap[random.randint(0, 1)])

    if genome[-1]:
        list_oob_score = [True, False]
        genome.append(list_oob_score[random.randint(0, 1)])
    else:
        genome.append(0)

    list_verbose = random.randint(1, 50)
    genome.append(list_verbose)

    list_warm_start = [True, False]
    genome.append(list_warm_start[random.randint(0, 1)])

    list_ccp_alpha = random.uniform(1, 10)
    genome.append(list_ccp_alpha)

    for i in range(0, number_of_features):
        genome.append(random.randint(0, 1))

    return icls(genome)


def mutation_rf(individual):
    number_parameter = random.randint(0, len(individual) - 1)

    if number_parameter == 0:
        individual[0] = random.randint(1, 5)
    if number_parameter == 1:
        list_criterion = ['gini', 'entropy', 'log_loss']
        individual[1] = list_criterion[random.randint(0, 2)]
    elif number_parameter == 2:
        individual[2] = random.randint(1, 600)
    elif number_parameter == 3:
        individual[3] = random.randint(2, 30)
    elif number_parameter == 4:
        individual[4] = random.randint(1, 30)
    elif number_parameter == 5:
        individual[5] = random.uniform(0.1, 0.5)
    elif number_parameter == 6:
        list_max_features = ['sqrt', 'log2', random.uniform(0.1, 1), random.randint(1, 13)]
        individual[6] = list_max_features[random.randint(0, 3)]
    elif number_parameter == 7:
        individual[7] = random.randint(2, 200)
    elif number_parameter == 8:
        individual[8] = random.uniform(1, 50)
    elif number_parameter == 9:
        list_bootstrap = [True, False]
        individual[9] = list_bootstrap[random.randint(0, 1)]
    elif number_parameter == 10:
        if individual[9]:
            list_oob_score = [True, False]
            individual[10] = list_oob_score[random.randint(0, 1)]
        else:
            individual[10] = 0
    elif number_parameter == 11:
        individual[11] = random.randint(1, 50)
    elif number_parameter == 12:
        list_warm_start = [True, False]
        individual[12] = list_warm_start[random.randint(0, 1)]
    elif number_parameter == 13:
        individual[13] = random.uniform(1, 10)



def mutation_rf_with_selection(individual):
    number_parameter = random.randint(0, len(individual) - 1)

    if number_parameter == 0:
        individual[0] = random.randint(1, 5)
    if number_parameter == 1:
        list_criterion = ['gini', 'entropy', 'log_loss']
        individual[1] = list_criterion[random.randint(0, 2)]
    elif number_parameter == 2:
        individual[2] = random.randint(1, 600)
    elif number_parameter == 3:
        individual[3] = random.randint(2, 30)
    elif number_parameter == 4:
        individual[4] = random.randint(1, 30)
    elif number_parameter == 5:
        individual[5] = random.uniform(0.1, 0.5)
    elif number_parameter == 6:
        list_max_features = ['sqrt', 'log2', random.uniform(0.1, 1), random.randint(1, 13)]
        individual[6] = list_max_features[random.randint(0, 3)]
    elif number_parameter == 7:
        individual[7] = random.randint(2, 200)
    elif number_parameter == 8:
        individual[8] = random.uniform(1, 50)
    elif number_parameter == 9:
        list_bootstrap = [True, False]
        individual[9] = list_bootstrap[random.randint(0, 1)]
    elif number_parameter == 10:
        if individual[9]:
            list_oob_score = [True, False]
            individual[10] = list_oob_score[random.randint(0, 1)]
        else:
            individual[10] = 0
    elif number_parameter == 11:
        individual[11] = random.randint(1, 50)
    elif number_parameter == 12:
        list_warm_start = [True, False]
        individual[12] = list_warm_start[random.randint(0, 1)]
    elif number_parameter == 13:
        individual[13] = random.uniform(1, 10)
    else:
        if individual[number_parameter] == 0:
            individual[number_parameter] = 1
        else:
            individual[number_parameter] = 0


def rf_parameter_feature_fitness(y, df, number_of_attributes, individual):
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

    estimator = RandomForestClassifier(
        n_estimators=individual[0],
        criterion=individual[1],
        max_depth=individual[2],
        min_samples_split=individual[3],
        min_samples_leaf=individual[4],
        min_weight_fraction_leaf=individual[5],
        max_features=individual[6],
        max_leaf_nodes=individual[7],
        min_impurity_decrease=individual[8],
        bootstrap=individual[9],
        # oob_score=individual[10],
        verbose=individual[11],
        warm_start=individual[12],
        ccp_alpha=individual[13],
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


class RfManager:

    def get_parameters_with_selection(self):
        return rf_parameters_with_selection

    def get_parameters(self):
        return rf_parameters

    def get_parameters_fitness(self):
        return rf_parameters_fitness

    def get_parameters_fitness_with_selection(self):
        return rf_parameter_feature_fitness

    def get_mutation(self):
        return mutation_rf

    def get_mutation_with_selection(self):
        return mutation_rf_with_selection

    def get_default_classifier(self):
        return RandomForestClassifier()
