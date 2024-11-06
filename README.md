## Rational
When splitting data between test and train splits, ideally the test and training sets should have the same distribution, to ensure that the test set is representative of the training set. However, Group Shuffle Split may lead to uneven distributions due to its random nature, affecting model evaluation.

When using Group Shuffle Split, some splits have similar distributions for test and training as seen below.

!["Close distributions"](/examples/Ideal_split.svg)

Because GroupShuffleSplit, is random, some splits will have a skewed distribution as shown below. This example illustrates the largest change in mean target observed in 10 splits using GroupShuffleSplit:

!["Large target shift"](/examples/Max_shift.svg)

To prevent this shift, the target variable can be stratified into bins and an equal number of test sets can be pulled from each bin, better preserving that target distribution.  The figure below shows the largest target shift for 10 splits with StratifiedGroupShuffleSplit

!["Largest stratified target shift"](/examples/stratified_max_shift.svg)


## Usage
StratifiedGroupShuffleSplit can be used to replace GroupShuffleSplit

```python
from stratified_group_shuffle_split import StratifiedGroupShuffleSplit
import numpy as np
from sklearn.datasets import make_regression

# Generate synthetic data
x, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
groups = np.arange(100)  # Dummy group labels

# Set up StratifiedGroupShuffleSplit
stratified_split = StratifiedGroupShuffleSplit(n_splits=10, test_size=0.2, n_bins=10, random_state=42)

# Iterate over splits
for train_idx, test_idx in stratified_split.split(x, y, groups):
    print("Train indices:", train_idx)
    print("Test indices:", test_idx)
```


## Future Ideas
# Fixes
Fix the logic to keep the test size close to the specified result for any test_size, n_bin combination. Currently they have to be specified to work correctly

## Contributing
We welcome contributions to enhance this project. Please open an issue or submit a pull request for any improvements.

## License
This project is licensed under the MIT License.
