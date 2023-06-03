from sklearn.preprocessing import MinMaxScaler
import random
from util import get_training_results
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.naive_bayes import ComplementNB
from util import get_training_results


def nb_parameters(number_of_features, icls):
    genome = list()

    list_alpha = random.uniform(0, 100)
    genome.append(list_alpha)

    list_force_alpha = [True, False]
    genome.append(list_force_alpha[random.randint(0, 1)])

    list_fit_prior = [True, False]
    genome.append(list_fit_prior[random.randint(0, 1)])

    list_norm = [True, False]
    genome.append(list_norm[random.randint(0, 1)])

    return icls(genome)


def nb_parameters_fitness(y, df, number_of_attributes, individual):
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    estimator = ComplementNB(
        alpha = individual[0],
        force_alpha = individual[1],
        fit_prior = individual[2],
        norm = individual[2]
    )

    return get_training_results(df_norm, y, estimator)


def nb_parameters_with_selection(number_of_features, icls):
    genome = list()

    list_alpha = random.uniform(0, 100)
    genome.append(list_alpha)

    list_force_alpha = [True, False]
    genome.append(list_force_alpha[random.randint(0, 1)])

    list_fit_prior = [True, False]
    genome.append(list_fit_prior[random.randint(0, 1)])

    list_norm = [True, False]
    genome.append(list_norm[random.randint(0, 1)])

    for i in range(0, number_of_features):
        genome.append(random.randint(0, 1))

    return icls(genome)


def mutation_nb(individual):
    number_parameter = random.randint(0, len(individual) - 1)

    if number_parameter == 0:
        individual[0] = random.uniform(0, 100)
    elif number_parameter == 1:
        list_force_alpha = [True, False]
        individual[1] = list_force_alpha[random.randint(0, 1)]
    elif number_parameter == 2:
        list_fit_prior = [True, False]
        individual[2] = list_fit_prior[random.randint(0, 1)]
    elif number_parameter == 3:
        list_norm = [True, False]
        individual[3] = list_norm[random.randint(0, 1)]


def mutation_nb_with_selection(individual):
    number_parameter = random.randint(0, len(individual) - 1)

    if number_parameter == 0:
        individual[0] = random.uniform(0, 100)
    elif number_parameter == 1:
        list_force_alpha = [True, False]
        individual[1] = list_force_alpha[random.randint(0, 1)]
    elif number_parameter == 2:
        list_fit_prior = [True, False]
        individual[2] = list_fit_prior[random.randint(0, 1)]
    elif number_parameter == 3:
        list_norm = [True, False]
        individual[3] = list_norm[random.randint(0, 1)]
    else:
        if individual[number_parameter] == 0:
            individual[number_parameter] = 1
        else:
            individual[number_parameter] = 0


def nb_parameter_feature_fitness(y, df, number_of_attributes, individual):
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

    estimator = ComplementNB(
        alpha = individual[0],
        force_alpha = individual[1],
        fit_prior = individual[2],
        norm = individual[2]
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


class nbManager:

    def get_parameters_with_selection(self):
        return nb_parameters_with_selection

    def get_parameters(self):
        return nb_parameters

    def get_parameters_fitness(self):
        return nb_parameters_fitness

    def get_parameters_fitness_with_selection(self):
        return nb_parameter_feature_fitness

    def get_mutation(self):
        return mutation_nb

    def get_mutation_with_selection(self):
        return mutation_nb_with_selection

    def get_default_classifier(self):
        return ComplementNB()
