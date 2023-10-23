import numpy as np
from sklearn.linear_model import LinearRegression

def flatten(arr):
    """
    Flatten a NumPy array.
    
    Args:
        arr (np.array): Input NumPy array.

    Returns:
        np.array: Flattened array.
    """
    return arr.ravel()

def revert_flatten(arr, figsize=(461, 475)):
    """
    Reshape a NumPy array to its original shape.

    Args:
        arr (np.array): Input NumPy array.
        figsize (tuple): Shape of the original array.

    Returns:
        np.array: Reshaped array.
    """
    return arr.reshape(figsize)

def create_dataset(X, y):
    """
    Create a dataset for linear regression.

    Args:
        X (list or np.array): Input feature array.
        y (list or np.array): Target array.

    Returns:
        X_train (np.array): Reshaped feature array.
        y_train (np.array): Flattened target array.
    """
    X_train = np.array(X).reshape(-1, 1)
    y_train = np.array([flatten(x) for x in y])

    return X_train, y_train

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
    
    def train(self, X, y):
        """
        Train the linear regression model.

        Args:
            X (np.array): Input feature array.
            y (np.array): Target array.
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Predict values using the trained model.

        Args:
            X (float): Input feature value (rain rate).

        Returns:
            np.array: Predicted target values.
        """
        assert X > 0, "Rain rate must be greater than 0"
        if X < 150:
            return np.zeros((461, 475))

        X_test = np.array([[X]])
        y_pred = self.model.predict(X_test)

        # Ensure predictions are non-negative
        y_pred_mod = np.array([max(0, x) for x in y_pred[0]])
        return revert_flatten(y_pred_mod)
    
    def score(self, X, y):
        """
        Calculate the R-squared score of the model.

        Args:
            X (np.array): Input feature array.
            y (np.array): Target array.

        Returns:
            float: R-squared score.
        """
        return self.model.score(X, y)