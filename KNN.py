import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import random
from sklearnex.neighbors import KNeighborsClassifier
from util import get_training_results


def knn_parameters(number_of_features, icls):
    genome = list()

    n_neighbors = random.randint(1, 10)
    genome.append(n_neighbors)

    list_weights = ['uniform', 'distance']
    genome.append(list_weights[random.randint(0, 1)])

    list_algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    genome.append(list_algorithm[random.randint(0, 3)])

    leaf_size = random.randint(10, 50)
    genome.append(leaf_size)

    p = random.choice([1, 2])
    genome.append(p)

    list_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    genome.append(list_metrics[random.choice([0, 1, 2, 3])])

    return icls(genome)


def knn_parameters_fitness(y, df, number_of_attributes, individual):
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    estimator = KNeighborsClassifier(
        n_neighbors=individual[0],
        weights=individual[1],
        algorithm=individual[2],
        leaf_size=individual[3],
        p=individual[4],
        metric=individual[5]
    )

    return get_training_results(df_norm, y, estimator)


def knn_parameters_with_selection(number_of_features, icls):
    genome = list()

    n_neighbors = random.randint(1, 10)
    genome.append(n_neighbors)

    list_weights = ['uniform', 'distance']
    genome.append(list_weights[random.randint(0, 1)])

    list_algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    genome.append(list_algorithm[random.randint(0, 3)])

    leaf_size = random.randint(10, 50)
    genome.append(leaf_size)

    p = random.choice([1, 2])
    genome.append(p)

    list_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    genome.append(list_metrics[random.choice([0, 1, 2, 3])])

    for i in range(0, number_of_features):
        genome.append(random.randint(0, 1))

    return icls(genome)


def mutation_knn(individual):
    number_parameter = random.randint(0, len(individual) - 1)

    if number_parameter == 0:
        individual[0] = random.randint(1, 10)
    elif number_parameter == 1:
        list_weights = ['uniform', 'distance']
        individual[1] = list_weights[random.randint(0, 1)]
    elif number_parameter == 2:
        list_algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
        individual[2] = list_algorithm[random.randint(0, 3)]
    elif number_parameter == 3:
        individual[3] = random.randint(10, 50)
    elif number_parameter == 4:
        individual[4] = random.choice([1, 2])
    elif number_parameter == 5:
        list_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        individual[5] = list_metrics[random.randint(0, 3)]

def mutation_knn_with_selection(individual):
    number_parameter = random.randint(0, len(individual) - 1)

    if number_parameter == 0:
        individual[0] = random.randint(1, 10)
    elif number_parameter == 1:
        list_weights = ['uniform', 'distance']
        individual[1] = list_weights[random.randint(0, 1)]
    elif number_parameter == 2:
        list_algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
        individual[2] = list_algorithm[random.randint(0, 3)]
    elif number_parameter == 3:
        individual[3] = random.randint(10, 50)
    elif number_parameter == 4:
        individual[4] = random.choice([1, 2])
    elif number_parameter == 5:
        list_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        individual[5] = list_metrics[random.randint(0, 3)]
    else:
        if individual[number_parameter] == 0:
            individual[number_parameter] = 1
        else:
            individual[number_parameter] = 0

def knn_parameter_feature_fitness(y, df, number_of_attributes, individual):
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

    estimator = KNeighborsClassifier(
        n_neighbors=individual[0],
        weights=individual[1],
        algorithm=individual[2],
        leaf_size=individual[3],
        p=individual[4],
        metric=individual[5]
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


class KnnManager:

    def get_parameters_with_selection(self):
        return knn_parameters_with_selection

    def get_parameters(self):
        return knn_parameters

    def get_parameters_fitness(self):
        return knn_parameters_fitness

    def get_parameters_fitness_with_selection(self):
        return knn_parameter_feature_fitness

    def get_mutation(self):
        return mutation_knn

    def get_mutation_with_selection(self):
        return mutation_knn_with_selection

    def get_default_classifier(self):
        return KNeighborsClassifier()
