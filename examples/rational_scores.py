# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>
"""
Module to generate and compare R² scores for ShuffleSplit
and StratifiedGroupShuffleSplit strategies.
The module generates synthetic regression data, applies two different split methods (ShuffleSplit
and StratifiedGroupShuffleSplit), and plots a bar chart comparing the R² scores of each.

Functions:
    get_train_test_scores: Perform cross-validation and calculate R² scores (mean and std)
    for train and test sets.
    plot_comparison_scores: Generate a bar chart comparing R² scores across split strategies.

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit, cross_validate
from stratified_group_shuffle_split import StratifiedGroupShuffleSplit

# Configuration
FILENAME = "Rational_scores.svg"
RANDOM_SEED = 42
NOISE = 10  # Of the synthetic data


def get_train_test_scores(_x, _y, splitter, _groups=None):
    """
    Calculate the R² score for train and test sets using cross-validation.

    Parameters:
        _x (array): Feature array.
        _y (array): Target array.
        splitter: Cross-validation splitting strategy (ShuffleSplit or StratifiedGroupShuffleSplit).
        _groups (array, optional): Group labels for StratifiedGroupShuffleSplit. Default is None.

    Returns:
        tuple: Means and standard deviations of train and test R² scores.
    """
    model = LinearRegression()

    # Perform cross-validation to get both train and test scores
    scores = cross_validate(model, _x, _y, cv=splitter.split(
        _x, _y, groups=_groups) if _groups is not None else splitter,
        scoring='r2', return_train_score=True)

    # Calculate mean and standard deviation for R² scores
    train_mean, train_std = np.mean(scores['train_score']), np.std(scores['train_score'])
    test_mean, test_std = np.mean(scores['test_score']), np.std(scores['test_score'])

    return train_mean, train_std, test_mean, test_std


def plot_comparison_scores(shuffle_scores, stratified_scores):
    """
    Plot a bar chart comparing R² scores for ShuffleSplit and StratifiedGroupShuffleSplit.

    Parameters:
        shuffle_scores (tuple): Train and test R² scores (mean and std) for ShuffleSplit.
        stratified_scores (tuple): Train and test R² scores (mean and std)
        for StratifiedGroupShuffleSplit.
    """
    labels = ['Shuffle Train R2', 'Stratified Shuffle Train R2', 'Shuffle Test R2',
              'Stratified Shuffle Test R2']
    means = [shuffle_scores[0], stratified_scores[0], shuffle_scores[2], stratified_scores[2]]
    stds = [shuffle_scores[1], stratified_scores[1], shuffle_scores[3], stratified_scores[3]]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, means, yerr=stds, capsize=10, color=['blue', 'blue', 'red', 'red'], alpha=0.6)
    ax.set_ylabel("R² Score")
    ax.set_title("R² Score Comparison")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.ylim([0.90, 1.0])

    # Save the figure to a file
    plt.savefig(FILENAME, format='svg')
    plt.show()


# Generate synthetic data
x, y = make_regression(n_samples=100, n_features=1, noise=NOISE, random_state=RANDOM_SEED)

# Define the split strategies
shuffle_split = ShuffleSplit(n_splits=10, test_size=0.2, random_state=RANDOM_SEED)
groups = np.arange(len(y))  # Groups are unique labels here for simplicity
stratified_split = StratifiedGroupShuffleSplit(n_splits=10, test_size=0.2, n_bins=10,
                                               random_state=RANDOM_SEED)

# Calculate scores for each strategy
shuffle_scores_ = get_train_test_scores(x, y, shuffle_split)
stratified_scores_ = get_train_test_scores(x, y, stratified_split, _groups=groups)

# Plot comparison of scores and save the figure
plot_comparison_scores(shuffle_scores_, stratified_scores_)
