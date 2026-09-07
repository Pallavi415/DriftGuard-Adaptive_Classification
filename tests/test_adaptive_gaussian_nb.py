import numpy as np

from driftaware.models.adaptive_gaussian_nb import AdaptiveGaussianNB


def test_adaptive_gaussian_nb_updates_and_predicts():
    model = AdaptiveGaussianNB(n_features=2)

    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [1.0, 1.0],
        [1.1, 1.1]
    ])

    y = np.array([0, 0, 1, 1])

    # Update the model with training observations
    model.update(X, y)

    # Check that predictions are produced
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_adaptive_gaussian_nb_decay():
    model = AdaptiveGaussianNB(
        n_features=2,
        decay_factor=0.5
    )

    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0]
    ])

    y = np.array([0, 1])

    # Build initial statistics
    model.update(X, y)

    original_counts = model.class_counts.copy()

    # Apply adaptation
    model.adapt()

    # Verify that older statistics were reduced
    assert np.allclose(
        model.class_counts,
        original_counts * 0.5
    )