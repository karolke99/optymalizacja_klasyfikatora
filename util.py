import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


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


def get_training_results(df_norm, y, estimator):
    split = 5
    cv = StratifiedKFold(n_splits=split)
    result_sum = 0

    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()

        result = (tp + tn) / (tp + fp + tn + fn)
        result_sum = result_sum + result

    return result_sum / split,
