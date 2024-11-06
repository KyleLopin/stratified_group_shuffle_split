
""" Not set for use, the StratifiedGroupShuffleSplit needs better logic to control test sizes
when a test_size and bin number are not set properly """

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.metrics import r2_score
from stratified_group_shuffle_split import StratifiedGroupShuffleSplit

RANDOM_SEED = 42


def learning_curve_splitter(x, y, splitter, groups=None):
    model = LinearRegression()
    train_sizes = [.6, .7, .8, .9, 1.0]

    train_scores = []
    test_scores = []

    # Loop over training sizes
    for train_size in train_sizes:
        n_train = int(train_size * len(y))  # Calculate number of training samples

        # Shuffle and split data based on the current train size
        if groups is not None:
            scores = cross_validate(model, x[:n_train], y[:n_train],
                                    cv=splitter.split(x[:n_train], y[:n_train],
                                                      groups=groups[:n_train]),
                                    scoring='r2', return_train_score=True)
        else:
            scores = cross_validate(model, x[:n_train], y[:n_train],
                                    cv=splitter.split(x[:n_train], y[:n_train]),
                                    scoring='r2', return_train_score=True)

        # Append the mean train and test scores for this training size
        train_scores.append(np.mean(scores['train_score']))
        test_scores.append(np.mean(scores['test_score']))

    return train_sizes, train_scores, test_scores


def plot_learning_curves(shuffle_train_sizes, shuffle_train_scores, shuffle_test_scores,
                         stratified_train_sizes, stratified_train_scores, stratified_test_scores):
    plt.figure(figsize=(10, 6))

    # Plot learning curve for ShuffleSplit
    plt.plot(shuffle_train_sizes, shuffle_train_scores, label="ShuffleSplit Train", color="blue",
             marker="o")
    plt.plot(shuffle_train_sizes, shuffle_test_scores, label="ShuffleSplit Test", color="red",
             marker="o")

    # Plot learning curve for StratifiedGroupShuffleSplit
    plt.plot(stratified_train_sizes, stratified_train_scores,
             label="StratifiedGroupShuffleSplit Train", color="green", marker="o")
    plt.plot(stratified_train_sizes, stratified_test_scores,
             label="StratifiedGroupShuffleSplit Test", color="purple", marker="o")

    plt.title("Learning Curves: ShuffleSplit vs StratifiedGroupShuffleSplit")
    plt.xlabel("Training Size")
    plt.ylabel("R² Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Generate synthetic data
x, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=RANDOM_SEED)

# Define the split strategies
shuffle_split = ShuffleSplit(n_splits=10, test_size=0.2, random_state=RANDOM_SEED)
# Create dummy group labels for StratifiedGroupShuffleSplit
groups = np.arange(len(y))  # Groups are just unique labels here for simplicity
stratified_split = StratifiedGroupShuffleSplit(n_splits=10, test_size=0.2, n_bins=10,
                                               random_state=RANDOM_SEED)

# Get the learning curves for both splitters
shuffle_train_sizes, shuffle_train_scores, shuffle_test_scores = learning_curve_splitter(x, y,
                                                                                         shuffle_split)
stratified_train_sizes, stratified_train_scores, stratified_test_scores = learning_curve_splitter(x,
                                                                                                  y,
                                                                                                  stratified_split,
                                                                                                  groups=groups)

# Plot the learning curves for both
plot_learning_curves(shuffle_train_sizes, shuffle_train_scores, shuffle_test_scores,
                     stratified_train_sizes, stratified_train_scores, stratified_test_scores)
