"""
Note on this code: this represents the implementaion of the simple linear regression
algorithm we saw during lectures 1 and 2, as implemented in the latter.
During assignment 1, you will have to implement the multiple linear regression algorithm
in a way that (1) there is no default value for the intercept and slope, and (2) the training
data and ground truth are passed as arguments to the fit function, not during initialization.
This file additionally includes an example of how to use the class (in the main).
You can run this script by calling `python simple_linear_regressor.py` AFTER having activated
your uv environment. Optionally, you can call `uv run simple_linear_regressor.py`.
"""

import numpy as np
from matplotlib import pyplot as plt


class SimpleLinearRegressor:
    def __init__(
        self,
        data: np.ndarray,
        ground_truth: np.ndarray,
        default_intercept: float = 0.0,
        default_slope: float = 0.0,
    ) -> None:
        """
        Initializes a linear regression model.

        Args:
            data (np.ndarray): Input data.
            ground_truth (np.ndarray): Corresponding ground truth.
            default_intercept (float, optional): Initial intercept value. Defaults to 0.0.
            default_slope (float, optional): Initial slope value. Defaults to 0.0.

        Attributes:
            _intercept (float): Initial intercept value.
            _slope (float): Initial slope value.
            _data (np.ndarray): Input data.
            _ground_truth (np.ndarray): Corresponding ground truth.
        """
        self._intercept: float = default_intercept
        self._slope: float = default_slope
        self._data = data
        self._ground_truth = ground_truth

    def fit(self) -> None:
        """
        Calculates the intercept and slope of the best fit line.

        The algorithm used is ordinary least squares. The best fit line is calculated
        by finding the values of intercept and slope that minimizes the sum of the
        squares of the vertical distances between each data point and the line.
        """
        x_bar = self._data.mean()
        y_bar = self._ground_truth.mean()
        x_deviance = self._data - x_bar
        y_deviance = self._ground_truth - y_bar
        numerator = (x_deviance * y_deviance).sum()
        denominator = (x_deviance**2).sum()
        self.slope = numerator / denominator

        self.intercept = y_bar - self.slope * x_bar

    def predict(self, new_data: np.ndarray) -> np.ndarray:
        """
        Predicts the output values for the given input data.

        Args:
            new_data (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted output values.
        """
        return self.intercept + self.slope * new_data


if __name__ == "__main__":
    data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ground_truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    # add a bit of noise to the x axis
    data += np.random.normal(0, 1, size=data.shape)

    plt.scatter(data, ground_truth)
    plt.title("The dataset")
    plt.show()

    regressor = SimpleLinearRegressor(data, ground_truth)
    regressor.fit()

    # create a dense sequence of x values for plotting line
    x = np.linspace(min(data), max(data), 1000)
    y = regressor.predict(x)

    plt.scatter(data, ground_truth)
    plt.plot(x, y, c="red")
    plt.title("The best fit line")
    plt.show()

    new_data = np.random.rand(5) * 10
    predicted = regressor.predict(data)

    plt.plot(x, y, c="red")
    plt.scatter(data, predicted, c="black")
    plt.title("Prediction on new data (matches line)")
    plt.show()
