## Rational
When splitting data between test and train splits, ideally the test and training sets should have the distribution

When using Group Shuffle Split, some splits have similar distributions for test and training as seen below.

!["Close distributions"](/examples/Ideal_split.svg)

When using GroupShuffleSplit, because it is random, some splits will have a skewed distribution as shown below.

!["Large target shift"](/examples/Max_shift.svg)

This is the largest change in the mean target found in 10 splits from GroupShuffleSplit split.

To prevent this shift the target variable can be stratified into bins and an equal number of test sets can be pulled from each bin, more closely preserving that target distribution.  The figure below shows the largest target shift for 10 splits with StratifiedGroupShuffleSplit

!["Largest stratified target shift"](/examples/stratified_max_shift.svg)


## Usage
StratifiedGroupShuffleSplit can be used to replace GroupShuffleSplit


## Future Ideas
# Fixes
Fix the logic to keep the test size close to the specified result for any test_size, n_bin combination. Currently they have to be specified to work correctly

## Contributing
We welcome contributions to enhance this project. Please open an issue or submit a pull request for any improvements.

## License
This project is licensed under the MIT License.
