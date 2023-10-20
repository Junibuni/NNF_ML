import numpy as np

from sklearn.linear_model import LinearRegression

def _flatten(arr):
    return np.array(arr.ravel())

def _revert_flatten(arr, figsize=(461,475)):
    return np.array(arr.reshape(figsize))

def create_dataset(X, y):
    X_train = np.array(X) #list shape (Xtrain, 1)
    y_train = np.array(y) #list shape (ytrain, y.shape[0], y.shape[1])

    X_train = X_train.reshape(-1, 1)
    y_train = np.array(list(map(lambda x:_flatten(x), y_train)))

    return X_train, y_train

class LR:
    def __init__(self):
        self.model = LinearRegression()
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        assert X > 0, "rain rate must be greater than 0"
        X_test = np.array([[X]])
        y_pred = self.model.predict(X_test)

        y_pred_mod = np.array([max(0, x) for x in y_pred[0]])
        return _revert_flatten(y_pred_mod)
    
    def score(self, x, y):
        return self.model.score(x, y)