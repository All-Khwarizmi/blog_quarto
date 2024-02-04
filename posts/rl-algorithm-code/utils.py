import math
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


def standardize_data(X_train, y_train):
    """Function to standardize the data using the mean and standard deviation
    Args:
        X_train (numpy array): The input data for training
        y_train (numpy array): The target data for training
    Returns:
        X_train_norm (numpy array): The normalised input data for training
        y_train_norm (numpy array): The normalised target data for training
    """
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    y_mean = np.mean(y_train, axis=0)
    y_std = np.std(y_train, axis=0)
    X_train_norm = (X_train - X_mean) / X_std
    y_train_norm = (y_train - y_mean) / y_std
    return X_train_norm, y_train_norm


class LinearRegression:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.m = X_train.shape[0]
        self.n = X_train.shape[1]
        self.b = np.ones((self.m, 1))
        self.w = np.zeros((self.n, 1))

    def forward_pass(self):

        fw_b = np.dot(self.X_train, self.w) + self.b
        return fw_b

    def cost_function(self, fw_b):
        cost = (1/(2*self.m)) * np.sum(np.square(fw_b - self.y_train))
        return cost

    def backward_pass(self, fw_b):
        dw = (1/self.m) * np.dot(self.X_train.T, (fw_b - self.y_train))
        db = (1/self.m) * np.sum(fw_b - self.y_train)
        return dw, db

    def update_parameters(self, dw, db, learning_rate):
        self.w = self.w - learning_rate * dw
        self.b = self.b - learning_rate * db

    def train(self, learning_rate=0.01, epochs=1000):
        costs = []
        for i in range(epochs):
            fw_b = self.forward_pass()
            cost = self.cost_function(fw_b)
            dw, db = self.backward_pass(fw_b)
            self.update_parameters(dw, db, learning_rate)
            costs.append(cost)
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
        return costs

    def predict(self, X_val):
        return np.dot(X_val, self.w) + self.b

    def plot_cost(self, costs):
        plt.plot(costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost function over iterations')
        plt.show()
