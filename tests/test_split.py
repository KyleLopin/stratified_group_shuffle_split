# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

import unittest
import numpy as np
import pandas as pd
from stratified_group_shuffle_split import StratifiedGroupShuffleSplit


class TestBasics(unittest.TestCase):
    """Tests for basic functionality of StratifiedGroupShuffleSplit"""

    def setUp(self):
        """Set up sample data for testing."""
        self.x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]])
        self.y = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
        self.groups = np.array(['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f'])

        # Default configuration
        self.spl = StratifiedGroupShuffleSplit(n_splits=1, test_size=0.35,
                                               n_bins=2, random_state=42)

    def test_split_output_shapes(self):
        """Test that the split method produces expected output shapes."""
        for train_idx, test_idx in self.spl.split(self.x, self.y, self.groups):
            self.assertEqual(len(train_idx) + len(test_idx), len(self.x))
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)

    def test_stratification(self):
        """Test that stratification is applied to target bins."""
        for train_idx, test_idx in self.spl.split(self.x, self.y, self.groups):
            train_y = self.y[train_idx]
            test_y = self.y[test_idx]
            train_bins = pd.qcut(train_y, q=self.spl.n_bins, labels=False, duplicates='drop')
            test_bins = pd.qcut(test_y, q=self.spl.n_bins, labels=False, duplicates='drop')

            # Check that each bin is represented in both train and test
            self.assertTrue(set(train_bins).issubset(set(range(self.spl.n_bins))))
            self.assertTrue(set(test_bins).issubset(set(range(self.spl.n_bins))))

    def test_random_state_reproducibility(self):
        """Test that setting random state produces reproducible splits."""
        spl1 = StratifiedGroupShuffleSplit(n_splits=1, test_size=0.35, n_bins=2, random_state=43)
        spl2 = StratifiedGroupShuffleSplit(n_splits=1, test_size=0.35, n_bins=2, random_state=43)

        train_test_pairs1 = list(spl1.split(self.x, self.y, self.groups))
        train_test_pairs2 = list(spl2.split(self.x, self.y, self.groups))
        print(train_test_pairs1)
        for (train_idx1, test_idx1), (train_idx2, test_idx2) in zip(train_test_pairs1,
                                                                    train_test_pairs2):
            np.testing.assert_array_equal(train_idx1, train_idx2)
            np.testing.assert_array_equal(test_idx1, test_idx2)


class TestExceptions(unittest.TestCase):
    """Tests for exception handling in StratifiedGroupShuffleSplit"""

    def setUp(self):
        """Set up sample data for testing."""
        self.X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        self.y = np.array([10, 20, 20, 30, 30, 40, 40, 50, 50, 60])
        self.groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    def test_missing_y(self):
        """Test that ValueError is raised when target y is missing."""
        spl = StratifiedGroupShuffleSplit()
        with self.assertRaises(ValueError):
            next(spl.split(self.X, groups=self.groups))

    def test_missing_groups(self):
        """Test that ValueError is raised when groups are missing."""
        spl = StratifiedGroupShuffleSplit()
        with self.assertRaises(ValueError):
            next(spl.split(self.X, y=self.y))

    def test_invalid_n_bins(self):
        """Test that ValueError is raised for invalid bin count."""
        with self.assertRaises(ValueError):
            StratifiedGroupShuffleSplit(n_bins=0)

    def test_invalid_test_size(self):
        """Test that ValueError is raised for test_size out of bounds."""
        with self.assertRaises(ValueError):
            StratifiedGroupShuffleSplit(test_size=1.5)


class TestDistribution(unittest.TestCase):
    def setUp(self):
        """Set up the test data and StratifiedGroupShuffleSplit instance."""
        # Generate data
        self.x = np.arange(300)  # Feature values [0, 1, 2, ..., 299]
        self.y = np.repeat(np.arange(100), 3)  # Target values [0, 0, 0, 1, 1, 1, ..., 99, 99, 99]
        self.groups = np.repeat(np.arange(100), 3)  # Group labels [0, 0, 0, 1, 1, 1, ..., 99, 99, 99]
        self.n_bins = 10
        # Initialize StratifiedGroupShuffleSplit with specified parameters
        self.spl = StratifiedGroupShuffleSplit(n_splits=20, test_size=0.1,
                                               n_bins=self.n_bins, random_state=42)

    def test_group_distribution_in_test_set(self):
        """Test that each split contains one group in each of the specified bins."""
        for train_idx, test_idx in self.spl.split(self.x, self.y, self.groups):
            # Initialize dictionaries for tracking bin counts for each category in train and test sets
            bins = {name: [0] * self.n_bins for name in
                    ['group_test', 'group_train', 'y_test', 'y_train', 'x_test', 'x_train',
                     'unique_group_test', 'unique_group_train']}

            # test each stratified text bin has just 1 group
            # calculate unique groups present in the test set for each split
            for test_group in np.unique(self.groups[test_idx]):
                # Determine the bin index (each bin represents a range of 10)
                bin_index = test_group // 10
                # Increment the count for the respective bin
                bins['unique_group_test'][bin_index] += 1
            for train_group in np.unique(self.groups[train_idx]):
                bins['unique_group_train'][train_group // 10] += 1

            # Loop over test indices and update bins for y_test, group_test, and x_test
            for test_idx_value in test_idx:
                # Determine the bin index for each category based on the y value
                bin_index = self.y[test_idx_value] // 10  # Determine bin index based on `y` value

                # Increment the bin count for y_test, group_test, and x_test
                bins['y_test'][bin_index] += 1
                bins['group_test'][self.groups[test_idx_value] // 10] += 1
                bins['x_test'][self.x[test_idx_value] // 30] += 1
            for train_idx_value in train_idx:
                # Increment the bin count for y_train, group_train, and x_train
                bins['y_train'][self.y[train_idx_value] // 10] += 1
                bins['group_train'][self.groups[train_idx_value] // 10] += 1
                bins['x_train'][self.x[train_idx_value] // 30] += 1
            expected_bins = {
                'group_test': [3] * 10,
                'group_train': [27] * 10,
                'y_test': [3] * 10,
                'y_train': [27] * 10,
                'x_test': [3] * 10,
                'x_train': [27] * 10,
                'unique_group_test': [1] * 10,
                'unique_group_train': [9] * 10
            }
            for key, expected_values in expected_bins.items():
                self.assertEqual(bins[key], expected_values,
                                 f"{key} counts do not match expected values")


if __name__ == "__main__":
    unittest.main()



