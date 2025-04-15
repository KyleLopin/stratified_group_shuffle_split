# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# local files
from stratified_group_shuffle_split.split import StratifiedGroupShuffleSplit


def visualize_split_distributions(x: pd.DataFrame,
                                  y: pd.Series,
                                  n_bins: int,
                                  title: str = None) -> None:
    """
    Perform a stratified group shuffle split on the target variable and visualize
    the distribution of the full, training, and testing targets.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target values. Used both for stratification and grouping.
    n_bins : int
        Number of bins to stratify the target variable.
    title : str, optional
        Title for the histogram and CDF plots.

    Returns
    -------
    None
        Displays the matplotlib plots directly.
    """
    # Perform stratified split internally
    sgss = StratifiedGroupShuffleSplit(n_splits=1, n_bins=n_bins,
                                       train_size=0.8, random_state=42)
    train_idx, test_idx = next(sgss.split(x=x, y=y, groups=y))

    y_train = y[train_idx]
    y_test = y[test_idx]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8),
                            gridspec_kw={'height_ratios': [3, 1]})

    # Compute common bin edges for consistent histogram comparison
    bin_edges = np.histogram_bin_edges(y, bins=n_bins)
    print(bin_edges)
    # Top plot: Histogram with KDE
    sns.histplot(y, bins=bin_edges, kde=True, color='black', label='Full', ax=axs[0], stat="density",
                 element="step")
    sns.histplot(y_train, bins=bin_edges, kde=True, color='blue', label='Train', ax=axs[0],
                 stat="density", element="step")
    sns.histplot(y_test, bins=bin_edges, kde=True, color='red', label='Test', ax=axs[0],
                 stat="density", element="step")

    axs[0].legend()
    axs[0].set_xlabel('Target Value')
    axs[0].set_ylabel('Density')
    if title:
        axs[0].set_title(title)

    # Bottom plot: CDF comparison
    for data, color, label in zip([y, y_train, y_test], ['black', 'blue', 'red'],
                                  ['Full', 'Train', 'Test']):
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axs[1].plot(sorted_data, cdf, label=label, color=color)

    axs[1].legend()
    axs[1].set_xlabel('Target Value')
    axs[1].set_ylabel('CDF')
    axs[1].set_title('Cumulative Distribution Functions')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    x_ = np.arange(1, 101)
    y_ = np.arange(1, 101)

    df = pd.DataFrame({
        'x': x_,
        'y': y_
    })
    visualize_split_distributions(df[['x']], df['y'] , 10)
