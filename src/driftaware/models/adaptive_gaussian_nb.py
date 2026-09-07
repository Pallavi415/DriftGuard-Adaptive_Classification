import numpy as np


class AdaptiveGaussianNB:
    """Incremental Gaussian Naive Bayes with drift-based statistic decay."""

    def __init__(self, n_features, epsilon=1e-9, decay_factor=0.5):
        self.n_features = n_features
        self.epsilon = epsilon
        self.decay_factor = decay_factor

        self.class_counts = np.zeros(2, dtype=float)
        self.means = np.zeros((2, n_features), dtype=float)
        self.m2 = np.zeros((2, n_features), dtype=float)

    def update(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Update the running statistics one observation at a time
        for row, target in zip(X, y):
            class_index = int(target)

            self.class_counts[class_index] += 1
            count = self.class_counts[class_index]

            difference = row - self.means[class_index]
            self.means[class_index] += difference / count

            updated_difference = row - self.means[class_index]
            self.m2[class_index] += difference * updated_difference

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        predictions = []

        # Calculate the most likely class for each observation
        for row in X:
            log_probabilities = []

            for class_index in range(2):
                if self.class_counts[class_index] == 0:
                    log_probabilities.append(-np.inf)
                    continue

                prior = self.class_counts[class_index] / self.class_counts.sum()

                variance = self.m2[class_index] / self.class_counts[class_index]
                variance = np.maximum(variance, self.epsilon)

                log_likelihood = -0.5 * np.sum(
                    np.log(2 * np.pi * variance)
                    + ((row - self.means[class_index]) ** 2) / variance
                )

                log_probabilities.append(
                    np.log(prior) + log_likelihood
                )

            predictions.append(np.argmax(log_probabilities))

        return np.array(predictions)

    def adapt(self):
        """Reduce the influence of older statistics after detected drift."""

        self.class_counts *= self.decay_factor
        self.m2 *= self.decay_factor