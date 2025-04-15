# Outline of features to add to StratifiedShuffleSplit

This document outlines features, improvements, and research directions for the project.  
Contributions and feedback are welcome.

## Current features to add
- Support **stratified-only splitting** without requiring `groups`
- Option to **return bin labels** (for analysis or reproducibility)
- Implement **StratifiedKFold** and **StratifiedGroupKFold** for cross-validation workflows
- Add **evaluation utilities** to score the split quality:
  - Mean/STD/Skewness comparison
- Make the functions work both **NumPy arrays and pandas DataFrames**

---

## Future extensions
- Allow binning on **derived features**, e.g.:
  - First principal component from PCA
  - Mean across multiple regression targets
- Add **visualization tools** for:
  - Per-bin balance heatmaps
  - Group distribution scatter plots
- Support **flexible binning strategies**:
  - Allow non-uniform binning methods (e.g., KMeans clustering on target values)
  - Let users pass custom bin edges or a binning function
- Make a **usage demonstrations**:
  - With end-to-end examples
  - Showcase learning curves, model fit quality, and how stratified splitting improves performance

---

## Rational - Research motivation
- Confirm and benchmark how **stratified splits** (vs. standard GroupShuffleSplit) impact:
  - Model performance
  - Generalization error
  - Learning curve convergence