# Import NumPy for expected statistical values
import numpy as np

# Import the running statistics implementation
from driftaware.models.running_stats import RunningGaussianStats


# Test Welford's algorithm against NumPy calculations
def test_running_statistics():
    # Create known values for validation
    values = np.array([10.0, 12.0, 14.0, 20.0])

    # Create the running statistics object
    stats = RunningGaussianStats()

    # Update the statistics one observation at a time
    for value in values:
        stats.update(value)

    # Check that the running mean matches NumPy
    assert np.isclose(stats.mean, np.mean(values))

    # Check that the running variance matches NumPy
    assert np.isclose(stats.variance, np.var(values))