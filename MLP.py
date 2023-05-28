from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import random
from util import get_training_results


def mlp_parameters(number_of_features, icls):
    genome = list()

    hidden_layer_sizes = random.randint(25, 75)
    genome.append(hidden_layer_sizes)

    list_activation = ['identity', 'logistic', 'tanh', 'relu']
    genome.append(list_activation[random.randint(0, 3)])

    list_solver = ['lbfgs', 'sgd', 'adam']
    genome.append(list_solver[random.randint(0, 2)])

    alpha = random.uniform(0.0001, 0.1)
    genome.append(alpha)

    batch_size = random.choice([8, 32, 64])
    genome.append(batch_size)

    list_learning_rate = ['constant', 'invscaling', 'adaptive']
    genome.append(list_learning_rate[random.randint(0, 2)])

    learning_rate_init = random.uniform(0.001, 0.1)
    genome.append(learning_rate_init)

    power_t = random.uniform(0, 1)
    genome.append(power_t)

    max_iter = random.randint(100, 500)
    genome.append(max_iter)

    shuffle = random.choice([True, False])
    genome.append(shuffle)

    tol = random.uniform(0.0001, 0.1)
    genome.append(tol)

    warm_start = random.choice([True, False])
    genome.append(warm_start)

    momentum = random.uniform(0, 1)
    genome.append(momentum)

    nesterovs_momentum = random.choice([True, False])
    genome.append(nesterovs_momentum)

    early_stopping = random.choice([True, False])
    genome.append(early_stopping)

    validation_fraction = random.uniform(0, 1)
    genome.append(validation_fraction)

    beta_1 = random.uniform(0, 1)
    genome.append(beta_1)

    beta_2 = random.uniform(0, 1)
    genome.append(beta_2)

    epsilon = random.uniform(0.00000001, 0.1)
    genome.append(epsilon)

    n_iter_no_change = random.randint(10, 30)
    genome.append(n_iter_no_change)

    return icls(genome)


def mlp_parameters_fitness(y, df, number_of_attributes, individual):
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    estimator = MLPClassifier(
        hidden_layer_sizes=individual[0],
        activation=individual[1],
        solver=individual[2],
        alpha=individual[3],
        batch_size=individual[4],
        learning_rate=individual[5],
        learning_rate_init=individual[6],
        power_t=individual[7],
        max_iter=individual[8],
        shuffle=individual[9],
        tol=individual[10],
        warm_start=individual[11],
        momentum=individual[12],
        nesterovs_momentum=individual[13],
        early_stopping=individual[14],
        validation_fraction=individual[15],
        beta_1=individual[16],
        beta_2=individual[17],
        epsilon=individual[18],
        n_iter_no_change=individual[19]
    )

    return get_training_results(df_norm, y, estimator)


def mutation_mlp(individual):
    number_parameter = random.randint(0, len(individual) - 1)

    if number_parameter == 0:
        individual[0] = random.randint(50, 150)
    elif number_parameter == 1:
        list_activation = ['identity', 'logistic', 'tanh', 'relu']
        individual[1] = list_activation[random.randint(0, 3)]
    elif number_parameter == 2:
        list_solver = ['lbfgs', 'sgd', 'adam']
        individual[2] = list_solver[random.randint(0, 2)]
    elif number_parameter == 3:
        individual[3] = random.uniform(0.0001, 0.1)
    elif number_parameter == 4:
        individual[4] = random.choice([32, 64, 128, 256])
    elif number_parameter == 5:
        list_learning_rate = ['constant', 'invscaling', 'adaptive']
        individual[5] = list_learning_rate[random.randint(0, 2)]
    elif number_parameter == 6:
        individual[6] = random.uniform(0.001, 0.1)
    elif number_parameter == 7:
        individual[7] = random.uniform(0, 1)
    elif number_parameter == 8:
        individual[8] = random.randint(100, 500)
    elif number_parameter == 9:
        individual[9] = random.choice([True, False])
    elif number_parameter == 10:
        individual[10] = random.uniform(0, 1)
    elif number_parameter == 11:
        individual[11] = random.choice([True, False])
    elif number_parameter == 12:
        individual[12] = random.uniform(0, 1)
    elif number_parameter == 13:
        individual[13] = random.choice([True, False])
    elif number_parameter == 14:
        individual[14] = random.choice([True, False])
    elif number_parameter == 15:
        individual[15] = random.uniform(0, 1)
    elif number_parameter == 16:
        individual[16] = random.uniform(0, 1)
    elif number_parameter == 17:
        individual[17] = random.uniform(0, 1)
    elif number_parameter == 18:
        individual[18] = random.uniform(0.00000001, 0.1)
    elif number_parameter == 19:
        individual[19] = random.randint(10, 30)


class MlpManager:

    def get_parameters(self):
        return mlp_parameters

    def get_parameters_fitness(self):
        return mlp_parameters_fitness

    def get_mutation(self):
        return mutation_mlp
