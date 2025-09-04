# Simple linear regression in Python - as seen during lecture

This repository contains the implementaion of the simple linear regression
algorithm we saw during lectures 1 and 2, as implemented in the latter.

Equation for optimal parameters according to Ordinary Least Squares (OLS):

$w^\star = \frac{\sum_{i=1}^n (x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^n (x_i-\bar{x})^2}$

$b^\star = \bar{y} - w^\star \cdot \bar{x}$

The prediction is run according to the line equation with the optimal parameters:

$\hat{y} = b^\star + w^\star x$

During assignment 1, you will have to implement the multiple linear regression algorithm
in a way that:
1. there is no default value for the intercept and slope, and
2. the training data and ground truth are passed as arguments to the fit function, not during initialization.

This file additionally includes an example of how to use the class (in the main).
You can run this script by calling `python simple_linear_regressor.py` AFTER having activated
your uv environment. Optionally, you can call `uv run simple_linear_regressor.py`.
Be careful about syncing the uv environment before running the code.