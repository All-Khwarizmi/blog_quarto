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


