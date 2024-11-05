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


if __name__ == "__main__":
    unittest.main()



