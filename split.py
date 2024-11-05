# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
stratified_group_shuffle_split.py

This module provides a custom cross-validator class `StratifiedGroupShuffleSplit`
for stratified sampling in regression tasks with grouped data. The class is designed
to act as a drop-in replacement for scikit-learn `GroupShuffleSplit` but adds support
for quantile-based binning of the target variable, allowing stratified splits for regression tasks.

Example usage:
    from stratified_group_shuffle_split import StratifiedGroupShuffleSplit
    splitter = StratifiedGroupShuffleSplit(n_splits=5, test_size=0.2, n_bins=5)
    for train_idx, test_idx in splitter.split(X, y, groups):
        # Train/test split your data based on the indices
"""

from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd


class StratifiedGroupShuffleSplit(BaseCrossValidator):
    """
    StratifiedGroupShuffleSplit cross-validator

    This class provides stratified sampling for regression tasks with grouped data.
    It uses quantile-based binning on the target variable, `y`, allowing the user
    to stratify based on target values while maintaining group integrity across
    train/test splits. It supports group-based stratification with an adjustable
    number of bins for stratification.

    Parameters
    ----------
    n_splits : int, default=1
        Number of re-shuffling & splitting iterations.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    train_size : float, optional
        Proportion of the dataset to include in the train split.
    random_state : int, RandomState instance, or None, optional
        Random state for reproducibility of splits.
    n_bins : int, default=5
        Number of quantile bins to split the target `y` for stratification.
    """

    def __init__(self, n_splits=1, test_size=0.2, train_size=None, random_state=None, n_bins=5):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.n_bins = n_bins

    def split(self, x, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target variable for regression tasks. Will be used to bin samples into quantile groups.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used to split the data.

        Yields
        ------
        train_idx : ndarray
            The training set indices for the current split.
        test_idx : ndarray
            The testing set indices for the current split.

        Raises
        ------
        ValueError
            If groups are not provided.
        """
        if y is None:
            raise ValueError("The target 'y' must be provided for stratification.")

        if groups is None:
            raise ValueError("Groups are required for StratifiedGroupShuffleSplit")

        # Bin the target variable based on quantiles
        y_binned = pd.qcut(y, q=self.n_bins, labels=False, duplicates='drop')

        # Create a DataFrame to combine groups and bins for stratification
        data = pd.DataFrame({'group': groups, 'y_binned': y_binned})

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_splits):
            # Split groups into stratified bins based on y_binned
            train_groups, test_groups = [], []

            for bin_value in data['y_binned'].unique():
                bin_groups = data[data['y_binned'] == bin_value]['group'].unique()
                rng.shuffle(bin_groups)

                n_test = int(len(bin_groups) * self.test_size)
                test_groups.extend(bin_groups[:n_test])
                train_groups.extend(bin_groups[n_test:])

            train_idx = data['group'].isin(train_groups).values
            test_idx = data['group'].isin(test_groups).values

            yield np.where(train_idx)[0], np.where(test_idx)[0]
