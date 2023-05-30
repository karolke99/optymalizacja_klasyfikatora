from sklearn.preprocessing import MinMaxScaler
import random
from util import get_training_results
from sklearn.linear_model import LogisticRegression


###### Logistic Regression ########

def lr_parameters(number_of_features, icls):
    genome = list()

    # c
    c = random.uniform(0.1, 100)
    genome.append(c)

    # solver
    solver_list = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
    solver = solver_list[random.randint(0, len(solver_list) - 1)]
    genome.append(solver)

    # penalty
    if solver == 'lbfgs':
        penalty_list = ['l2', 'none']
    elif solver == 'liblinear':
        penalty_list = ['l1', 'l2']
    elif solver == 'newton-cg':
        penalty_list = ['l2', 'none']
    elif solver == 'sag':
        penalty_list = ['l2', 'none']
    elif solver == 'saga':
        penalty_list = ['elasticnet', 'l1', 'l2', 'none']
    else:
        raise ValueError('Unsupported solver')

    penalty = penalty_list[random.randint(0, len(penalty_list) - 1)]
    genome.append(penalty)

    # li_ration only if elasticnet is selected
    if penalty == 'elasticnet':
        l1_ratio = random.uniform(0, 1)
    else:
        l1_ratio = None

    genome.append(l1_ratio)

    return icls(genome)


def lr_parameters_fitness(y, df, number_of_attributes, individual):
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    if individual[2] == 'elasticnet':
        estimator = LogisticRegression(
            C=individual[0],
            solver=individual[1],
            penalty=individual[2],
            l1_ratio=individual[3]
        )
    else:
        estimator = LogisticRegression(
            C=individual[0],
            solver=individual[1],
            penalty=individual[2],
        )

    return get_training_results(df_norm, y, estimator)


def mutation_lr(individual):
    number_parameter = random.randint(0, len(individual) - 2)

    if number_parameter == 0:
        individual[0] = random.uniform(0.1, 100)
    elif number_parameter == 1:
        solver_list = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
        solver = solver_list[random.randint(0, len(solver_list) - 1)]
        individual[1] = solver

        if solver == 'lbfgs':
            penalty_list = ['l2', 'none']
        elif solver == 'liblinear':
            penalty_list = ['l1', 'l2']
        elif solver == 'newton-cg':
            penalty_list = ['l2', 'none']
        elif solver == 'sag':
            penalty_list = ['l2', 'none']
        elif solver == 'saga':
            penalty_list = ['elasticnet', 'l1', 'l2', 'none']
        else:
            raise ValueError('Unsupported solver')

        penalty = penalty_list[random.randint(0, len(penalty_list) - 1)]
        individual[2] = penalty

        if penalty == 'elasticnet':
            l1_ratio = random.uniform(0, 1)
        else:
            l1_ratio = None

        individual[3] = l1_ratio

    elif number_parameter == 2:
        solver_list = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
        solver = solver_list[random.randint(0, len(solver_list) - 1)]
        individual[1] = solver

        if solver == 'lbfgs':
            penalty_list = ['l2', 'none']
        elif solver == 'liblinear':
            penalty_list = ['l1', 'l2']
        elif solver == 'newton-cg':
            penalty_list = ['l2', 'none']
        elif solver == 'sag':
            penalty_list = ['l2', 'none']
        elif solver == 'saga':
            penalty_list = ['elasticnet', 'l1', 'l2', 'none']
        else:
            raise ValueError('Unsupported solver')

        penalty = penalty_list[random.randint(0, len(penalty_list) - 1)]
        individual[2] = penalty

        if penalty == 'elasticnet':
            l1_ratio = random.uniform(0, 1)
        else:
            l1_ratio = None

        individual[3] = l1_ratio
    else:
        raise Exception('Wrong parameter to mutate')


class LrManager:

    def get_parameters(self):
        return lr_parameters

    def get_parameters_fitness(self):
        return lr_parameters_fitness

    def get_mutation(self):
        return mutation_lr
