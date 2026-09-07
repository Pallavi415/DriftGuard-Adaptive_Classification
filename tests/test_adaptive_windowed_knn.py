import numpy as np

from driftaware.models.adaptive_windowed_knn import AdaptiveWindowedKNN


def test_adaptive_windowed_knn_updates_and_predicts():
    model = AdaptiveWindowedKNN(
        window_size=4,
        n_neighbors=2
    )

    X = np.array([
        [0.0],
        [0.1],
        [1.0],
        [1.1]
    ])

    y = np.array([0, 0, 1, 1])

    # Add observations to the model
    model.update(X, y)

    # Check that predictions can be generated
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_adaptive_windowed_knn_reduces_window_after_drift():
    model = AdaptiveWindowedKNN(
        window_size=4,
        n_neighbors=2
    )

    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])

    y = np.array([0, 0, 1, 1])

    # Fill the normal KNN window
    model.update(X, y)

    # Simulate a detected change
    model.adapt()

    # Only the newest half should remain
    assert len(model.X_window) == 2
    assert np.array_equal(
        model.X_window.flatten(),
        np.array([3.0, 4.0])
    )