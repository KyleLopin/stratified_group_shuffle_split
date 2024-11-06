# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import ShuffleSplit

# local files
from stratified_group_shuffle_split import StratifiedGroupShuffleSplit

RANDOM_SEED = 42
NOISE = 10


def find_split_results(x, y, splitter, title_prefix, groups=None):
    # Variables to track the smallest and largest mean differences
    min_diff = float('inf')
    max_diff = float('-inf')
    min_diff_data = None
    max_diff_data = None
    min_diff_x = None
    max_diff_x = None
    groups = np.arange(100)  # Unique group labels for each sample to disable grouping

    # Iterate over each split to find the smallest and largest mean differences
    for train_idx, test_idx in splitter.split(x, y, groups):
        y_train, y_test = y[train_idx], y[test_idx]
        x_train, x_test = x[train_idx], x[test_idx]

        # Calculate the mean difference
        mean_diff = abs(np.mean(y_train) - np.mean(y_test))

        # Check if this is the smallest or largest difference encountered
        if mean_diff < min_diff:
            min_diff = mean_diff
            min_diff_data = (y_train, y_test)
            min_diff_x = (x_train, x_test)

        if mean_diff > max_diff:
            max_diff = mean_diff
            max_diff_data = (y_train, y_test)
            max_diff_x = (x_train, x_test)

    # Plot smallest mean difference
    plot_split_results(min_diff_x[0], min_diff_x[1], min_diff_data[0], min_diff_data[1],
                       f"{title_prefix}: Smallest Mean Difference (Diff = {min_diff:.2f})")

    # Plot largest mean difference
    plot_split_results(max_diff_x[0], max_diff_x[1], max_diff_data[0], max_diff_data[1],
                       f"{title_prefix}: Largest Mean Difference (Diff = {max_diff:.2f})")


def plot_split_results(x_train, x_test, y_train, y_test, title):
    # Create a single figure with 2 rows and 1 column
    fig, axes = plt.subplots(2, 1, figsize=(6, 10))

    # Plot y vs x scatter plots for the mean difference
    axes[0].scatter(y_train, x_train, color="blue", alpha=0.6, label="Train")
    axes[0].scatter(y_test, x_test, color="red", alpha=0.6, label="Test")
    axes[0].set_title(title)
    axes[0].set_xlabel("Target y")
    axes[0].set_ylabel("Feature X")
    axes[0].legend()

    # Histogram for mean difference
    axes[1].hist(y_train, bins=10, alpha=0.5, label="Train", color="blue")
    axes[1].hist(y_test, bins=10, alpha=0.5, label="Test", color="red")
    axes[1].axvline(np.mean(y_train), color='blue', linestyle='--', linewidth=1, label="Train Mean")
    axes[1].axvline(np.mean(y_test), color='red', linestyle='--', linewidth=1, label="Test Mean")
    axes[1].set_xlabel("y values")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("Hold.svg", format='svg')  # save the file each time, or it will overwrite
    plt.show()


# Generate dataset
x, y = make_regression(n_samples=100, n_features=1, noise=NOISE, random_state=RANDOM_SEED)

# Run for ShuffleSplit
shuffle_split = ShuffleSplit(n_splits=10, test_size=0.2, random_state=RANDOM_SEED)
find_split_results(x, y, shuffle_split, "ShuffleSplit")

# Run for StratifiedGroupShuffleSplit with groups
groups = np.arange(100)  # Unique group labels for each sample to disable grouping
stratified_group_split = StratifiedGroupShuffleSplit(n_splits=10, test_size=0.2, n_bins=10,
                                                     random_state=RANDOM_SEED)
find_split_results(x, y, stratified_group_split, "StratifiedGroupShuffleSplit")

# Additional Example for Group K-Fold Split if required:
# group_kfold = GroupKFold(n_splits=10)
# find_split_results(x, y, group_kfold, "GroupKFold", groups=groups)
