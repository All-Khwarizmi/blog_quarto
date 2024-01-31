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
    # Print the tye of X and y
    print(f"X type: {type(X)}")
    print(f"y type: {type(y)}")
    # Check if X and y are dataframes, if so convert them to numpy arrays
    X_np = X.copy()
    y_np = y.copy()
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X_np = X.to_numpy()
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y_np = y.to_numpy()
    print(f"X type: {type(X_np)}")
    print(f"y type: {type(y_np)}")
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


# Function to get parameters
def get_parameters(X):
    """Function to get the parameters of the model
    Args:
        X (numpy array): The input data
    Returns:
        W (numpy array): The weights
        b (float): The bias
    """
    n = 1
    # Check if the dimensions of X and create the params accordingly
    if len(X.shape) == 1:
        n = 1
    else:
        _, n = X.shape
    print("n shape = ", n)
    W = 1
    b = 1
    return W, b


# Function to get the predictions without dot product
def compute_model_output(X, W, b):
    """Function to get the predictions
    Args:
        x_i (float): The input data
        w_i (float): The weight
        b (float): The bias
    """
    m = X.shape[0]
    y_pred = np.zeros(m)
    for i in range(X.shape[0]):
        y_pred[i] = X[i] * W + b
    return y_pred

# Function to get the predictions


def predict_dot(X, W, b):
    """Function to get the predictions
    Args:
        X (numpy array): The input data
        W (numpy array): The weights
        b (float): The bias
    Returns:
        y_pred (numpy array): The predictions

    """
    y_pred = np.dot(X, W) + b
    return y_pred


# Cost function implementation
def fw_b(x_i, w_i, b):
    return x_i * w_i + b


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost


# Gradient descent
def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """

    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
        # Add print statement but only for the first 10 iterations
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient

    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    w = w_in
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        # Add print statement but only for the first 10 iterations

        # Save cost J at each iteration
        if i < 100000:      # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, J_history, p_history


# Function to normalize data
def normalize(x):
    """Function to normalize the data
    Args:
        x (numpy array): The input data
    Returns:
        x_norm (numpy array): The normalized data
    """
    m = x.shape[0]
    x_min = min(x)
    x_max = max(x)
    diff = x_max - x_min
    x_norm = np.zeros(m)
    for i in range(m):
        x_norm[i] = (x[i] - x_min) / diff

    return x_norm


# Metric function to measure the quality of the model
def mse(y_pred, y_labels):
    """Function to compute the mean squared error
    Args:
        y_pred (numpy array): The predictions
        y_labels (numpy array): The labels
    Returns:
        error_cost (float): The mean squared error
    """
    assert y_pred.shape == y_labels.shape
    
    m = y_labels.shape[0]
    
    error_cost = 0
    for pred in range(m):
        error_cost += (y_pred[pred] - y_labels[pred])**2
    
    error_cost / m
    
    return error_cost