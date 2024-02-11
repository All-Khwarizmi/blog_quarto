import math
from re import X
from matplotlib.pylab import f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Créer une fonction qui permet de séparer la donnée
# AKA sklearn.model_selection.train_test_split


def dataset_splitter(X, y, test_size=0.25, random_state=0, shuffle=True):
    """Function to split the dataset into train and test sets
    Args:
        X (numpy array): The input data
        y (numpy array): The target data
        test_size (float): The size of the test set
        random_state (int): The random state
        shuffle (bool): Whether to shuffle the data or not      
    Returns:
        X_train (numpy array): The input data for training
        X_val (numpy array): The input data for validation
        y_train (numpy array): The target data for training
        y_val (numpy array): The target data for validation
    """
    assert X.shape[0] == y.shape[0]
    # Check if test_size is a float between 0 and 1
    assert isinstance(test_size, float)
    assert test_size > 0 and test_size < 1
    # Check if random_state is an integer
    assert isinstance(random_state, int)
    # Check if arrays have a dimension greater than 1
    assert len(X.shape) > 0
    assert len(y.shape) > 0
    # Check if X and y are dataframes, if so convert them to numpy arrays
    X_np = X.copy()
    y_np = y.copy()
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X_np = X.to_numpy()
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y_np = y.to_numpy()

    # Shuffle the data
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(X_np)
        np.random.seed(random_state)
        np.random.shuffle(y_np)

    m = X.shape[0]
    slice_treshold = int((1-test_size)*m)
    X_train = X_np[:slice_treshold]
    X_val = X_np[slice_treshold:]
    y_train = y_np[:slice_treshold]
    y_val = y_np[slice_treshold:]

    print(f"train_size: {(1-test_size)*100}% - test_size: {test_size*100}%")
    print(f"X_train: {int(round(X_train.shape[0]/m*100))}%")
    print(f"X_val: {int(round(X_val.shape[0]/m*100))}%")

    return X_train, X_val, y_train, y_val


def standardize_data(X_train, X_test):
    """Function to standardize the data using the mean and standard deviation
    Args:
        X_train (numpy array): The input data for training
        y_train (numpy array): The target data for training
    Returns:
        X_train_norm (numpy array): The normalised input data for training
        y_train_norm (numpy array): The normalised target data for training
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test


class LinearRegression:
    """Class to implement a simple linear regression model
    Args:
        X_train (numpy array): The input data for training
        y_train (numpy array): The target data for training

    Attributes:
        X_train (numpy array): The input data for training
        y_train (numpy array): The target data for training
        m (int): The number of training examples
        n (int): The number of features
        b (float): The bias
        w (numpy array): The weights
    """

    def __init__(self, learning_rate, convergence_tol=1e-6):
        self.learning_rate = learning_rate
        self.convergence_tol = convergence_tol
        self.W = None
        self.b = None

    def initialize_parameters(self, n_features):
        """
        Initialize model parameters.

        Parameters:
            n_features (int): The number of features in the input data.
        """
        self.W = np.random.randn(n_features) * 0.01
        self.b = 0

    def forward(self, X):

        fw_b = np.dot(X, self.W) + self.b
        return fw_b

    def cost_function(self, fw_b):
        m = len(fw_b)
        cost = np.sum(np.square(fw_b - self.y) / (2 * m))
        return cost

    def backward(self, fw_b):
        m = len(fw_b)
        self.dW = np.dot((fw_b - self.y), self.X) / m
        self.db = np.sum(fw_b - self.y) / m

    def update_parameters(self):
        self.W = self.W - self.learning_rate * self.dW
        self.b = self.b - self.learning_rate * self.db

    def fit(self, X, y, iterations=1000):
        assert isinstance(X, np.ndarray), "X must be a NumPy array"
        assert isinstance(y, np.ndarray), "y must be a NumPy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert iterations > 0, "Iterations must be greater than 0"
        costs = []

        self.X = X
        self.y = y
        self.initialize_parameters(X.shape[1])

        for i in range(iterations):
            fw_b = self.forward(X)
            cost = self.cost_function(fw_b)
            self.backward(fw_b)
            self.update_parameters()
            costs.append(cost)
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
            if i > 0 and abs(costs[-1] - costs[-2]) < self.convergence_tol:
                print(f"Converged after {i} iterations.")
                break

        self.plot_cost(costs)

    def predict(self, X_val):
        return np.dot(X_val, self.W) + self.b

    def plot_cost(self, costs):
        plt.plot(costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost function over iterations')
        plt.show()

# A test to check if the linear regression model works


def test_linear_regression():
    X = np.array([[1, 2, 3, 4, 5]]).T
    y = np.array([[2, 4, 6, 8, 10]]).T
    X_train, X_val, y_train, y_val = dataset_splitter(X, y, test_size=0.2)
    X_train_norm, y_train_norm = standardize_data(X_train, y_train)
    model = LinearRegression(X_train_norm, y_train_norm)
    costs = model.fit(learning_rate=0.01, iterations=4000)
    model.plot_cost(costs)
    X_val_norm = (X_val - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    y_val_norm = (y_val - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
    y_pred = model.predict(X_val_norm)
    print(f"Predicted: {y_pred}")
    print(f"Actual: {y_val_norm}")
    assert math.isclose(y_pred[0][0], y_val_norm[0][0], rel_tol=0.1)
    assert math.isclose(y_pred[1][0], y_val_norm[1][0], rel_tol=0.1)
    assert math.isclose(y_pred[2][0], y_val_norm[2][0], rel_tol=0.1)
    assert math.isclose(y_pred[3][0], y_val_norm[3][0], rel_tol=0.1)
    assert math.isclose(y_pred[4][0], y_val_norm[4][0], rel_tol=0.1)
    print("All tests passed!")


if __name__ == "__main__":

    test_linear_regression()
