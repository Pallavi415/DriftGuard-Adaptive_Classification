# Import NumPy for creating test data
import numpy as np

# Import the windowed KNN implementation
from driftaware.models.windowed_knn import WindowedKNN


# Test that the KNN window keeps only recent observations
def test_window_size():
    # Create a windowed KNN with a small window for testing
    model = WindowedKNN(
        window_size=4,
        n_neighbors=1
    )

    # Create the first batch of observations
    X_first = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])

    # Create labels for the first batch
    y_first = np.array([0, 0, 1, 1])

    # Add the first batch to the model
    model.update(X_first, y_first)

    # Add new observations that exceed the window size
    X_second = np.array([
        [5.0],
        [6.0]
    ])

    # Create labels for the new observations
    y_second = np.array([1, 1])

    # Update the model with the new observations
    model.update(X_second, y_second)

    # Check that only the most recent four observations remain
    assert len(model.X_window) == 4

    # Check that the oldest observations were removed
    assert np.array_equal(
        model.X_window.flatten(),
        np.array([3.0, 4.0, 5.0, 6.0])
    )