import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class AdaptiveWindowedKNN:
    """Windowed KNN that removes older observations more aggressively after drift."""

    def __init__(self, window_size=2000, n_neighbors=5):
        self.window_size = window_size
        self.n_neighbors = n_neighbors
        self.X_window = None
        self.y_window = None

    def update(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Add the new observations to the current window
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

    def predict(self, X):
        if self.X_window is None or len(self.X_window) == 0:
            raise ValueError("The model has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        # Avoid using more neighbors than observations available
        neighbors = min(self.n_neighbors, len(self.X_window))

        # Fit KNN using the current recent-data window
        knn = KNeighborsClassifier(n_neighbors=neighbors)
        knn.fit(self.X_window, self.y_window)

        # Predict using the recent-data window
        return knn.predict(X)

    def adapt(self):
        """Remove older observations after a detected change."""

        if self.X_window is None:
            return

        # Keep only the most recent half of the current window
        new_window_size = self.window_size // 2

        self.X_window = self.X_window[-new_window_size:]
        self.y_window = self.y_window[-new_window_size:]