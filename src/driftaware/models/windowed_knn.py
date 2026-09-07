# Import NumPy for numerical operations
import numpy as np

# Import KNN from scikit-learn
from sklearn.neighbors import KNeighborsClassifier


class WindowedKNN:
    """KNN classifier that retains only a bounded window of recent observations."""

    # Initialize the KNN configuration
    def __init__(self, window_size=2000, n_neighbors=5):
        self.window_size = window_size
        self.n_neighbors = n_neighbors

        # Store the recent observations and their labels
        self.X_window = None
        self.y_window = None

    # Add new observations to the sliding window
    def update(self, X, y):
        # Convert the input data into NumPy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Add the new observations to the existing window
        if self.X_window is None:
            self.X_window = X.copy()
            self.y_window = y.copy()
        else:
            self.X_window = np.vstack([self.X_window, X])
            self.y_window = np.concatenate([self.y_window, y])

        # Keep only the most recent observations
        if len(self.X_window) > self.window_size:
            self.X_window = self.X_window[-self.window_size:]
            self.y_window = self.y_window[-self.window_size:]

    # Predict classes for new observations
    def predict(self, X):
        # Make sure the model has observations available
        if self.X_window is None or len(self.X_window) == 0:
            raise ValueError("The model has not been fitted yet.")

        # Convert the input data into NumPy arrays
        X = np.asarray(X, dtype=float)

        # Use no more neighbors than observations currently stored
        neighbors = min(self.n_neighbors, len(self.X_window))

        # Create a KNN classifier using the current window
        knn = KNeighborsClassifier(n_neighbors=neighbors)

        # Fit KNN on the current recent-history window
        knn.fit(self.X_window, self.y_window)

        # Generate predictions for the new observations
        return knn.predict(X)