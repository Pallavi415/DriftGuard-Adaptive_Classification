# Import NumPy for creating test data and comparisons
import numpy as np

# Import the incremental Gaussian Naive Bayes model
from driftaware.models.incremental_gaussian_nb import IncrementalGaussianNB


# Test that the model can learn from a small batch and make predictions
def test_incremental_gaussian_nb():
    # Create a small dataset with two clearly separated classes
    X = np.array([
        [1.0, 1.0],
        [1.2, 1.1],
        [5.0, 5.0],
        [5.2, 5.1]
    ])

    # Assign the first two observations to class 0
    # and the last two observations to class 1
    y = np.array([0, 0, 1, 1])

    # Create the incremental Gaussian Naive Bayes model
    model = IncrementalGaussianNB(n_features=2)

    # Update the model using the training observations
    model.update(X, y)

    # Create observations close to each class
    test_data = np.array([
        [1.1, 1.0],
        [5.1, 5.0]
    ])

    # Generate predictions
    predictions = model.predict(test_data)

    # Check that the predictions match the expected classes
    assert np.array_equal(predictions, np.array([0, 1]))