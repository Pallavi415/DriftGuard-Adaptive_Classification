# Import NumPy for numerical calculations
import numpy as np


class RunningGaussianStats:
    """Maintain running mean and variance using Welford's algorithm."""

    # Initialize the running statistics
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    # Update the statistics with one new observation
    def update(self, value):
        self.count += 1

        difference = value - self.mean
        self.mean += difference / self.count

        updated_difference = value - self.mean
        self.m2 += difference * updated_difference

    # Return the population variance
    @property
    def variance(self):
        if self.count == 0:
            return 0.0

        return self.m2 / self.count