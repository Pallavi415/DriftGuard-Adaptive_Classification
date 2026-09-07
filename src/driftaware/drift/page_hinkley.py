# Import NumPy for numerical calculations
import numpy as np


class PageHinkley:
    """Detect persistent changes in a streaming numeric signal."""

    # Initialize the Page-Hinkley detector
    def __init__(self, delta=0.005, threshold=50.0):
        self.delta = delta
        self.threshold = threshold

        # Initialize the running statistics
        self.count = 0
        self.mean = 0.0
        self.cumulative_sum = 0.0
        self.minimum_sum = 0.0

    # Reset the detector after a change has been detected
    def reset(self):
        # Start a new detection period
        self.count = 0
        self.mean = 0.0
        self.cumulative_sum = 0.0
        self.minimum_sum = 0.0

    # Update the detector with one new observation
    def update(self, value):
        # Convert the incoming value to a float
        value = float(value)

        # Update the number of observations
        self.count += 1

        # Update the running mean
        self.mean += (value - self.mean) / self.count

        # Update the cumulative deviation
        self.cumulative_sum += value - self.mean - self.delta

        # Track the minimum cumulative value
        self.minimum_sum = min(
            self.minimum_sum,
            self.cumulative_sum
        )

        # Measure the deviation from the minimum
        deviation = self.cumulative_sum - self.minimum_sum

        # Check whether the deviation exceeds the threshold
        if deviation > self.threshold:

            # Reset the detector so it can identify a future change
            self.reset()

            # Report that a change was detected
            return True

        # Report that no change was detected
        return False