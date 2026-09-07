# Import NumPy for numerical calculations
import numpy as np

# Import the running statistics used for each feature
from driftaware.models.running_stats import RunningGaussianStats


class IncrementalGaussianNB:
    """Gaussian Naive Bayes classifier that learns incrementally."""

    # Initialize the model configuration and statistics
    def __init__(self, n_features, classes=(0, 1), epsilon=1e-9):
        self.n_features = n_features
        self.classes = np.array(classes)
        self.epsilon = epsilon

        # Store the number of observations seen for each class
        self.class_counts = {
            class_label: 0 for class_label in self.classes
        }

        # Create running statistics for every class and feature
        self.feature_stats = {}

        for class_label in self.classes:
            self.feature_stats[class_label] = []

            for _ in range(self.n_features):
                self.feature_stats[class_label].append(
                    RunningGaussianStats()
                )

    # Update the model using a batch of observations
    def update(self, X, y):
        # Convert the input data into NumPy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Update the statistics one observation at a time
        for features, target in zip(X, y):
            self.class_counts[target] += 1

            # Update each feature's running statistics
            for feature_index, value in enumerate(features):
                self.feature_stats[target][feature_index].update(value)

    # Calculate the log probability of each class for one observation
    def _log_probability(self, features, class_label):
        # Get the number of observations belonging to this class
        class_count = self.class_counts[class_label]

        # Calculate the total number of observations seen by the model
        total_count = sum(self.class_counts.values())

        # Calculate the class prior probability
        class_prior = class_count / total_count

        # Start with the logarithm of the class prior
        log_probability = np.log(class_prior)

        # Add the Gaussian log-likelihood for every feature
        for feature_index, value in enumerate(features):
            stats = self.feature_stats[class_label][feature_index]

            # Keep the variance away from zero for numerical stability
            variance = max(stats.variance, self.epsilon)

            # Calculate the Gaussian log-likelihood
            log_likelihood = (
                -0.5 * np.log(2 * np.pi * variance)
                - ((value - stats.mean) ** 2) / (2 * variance)
            )

            # Add this feature's contribution to the class probability
            log_probability += log_likelihood

        return log_probability

    # Predict the class for each observation
    def predict(self, X):
        # Convert the input data into a NumPy array
        X = np.asarray(X, dtype=float)

        # Make sure the model has already seen some data
        if sum(self.class_counts.values()) == 0:
            raise ValueError("The model has not been fitted yet.")

        # Store the predictions
        predictions = []

        # Predict one observation at a time
        for features in X:
            class_probabilities = []

            # Calculate the log probability for every class
            for class_label in self.classes:
                if self.class_counts[class_label] == 0:
                    class_probabilities.append(-np.inf)
                else:
                    class_probabilities.append(
                        self._log_probability(features, class_label)
                    )

            # Select the class with the highest log probability
            predictions.append(
                self.classes[np.argmax(class_probabilities)]
            )

        return np.array(predictions)