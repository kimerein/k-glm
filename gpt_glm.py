# Import necessary modules
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold
import scipy
import numpy as np

# Define the data
#X = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
#y = [1, 2, 3, 4, 5]

# Get the data from Matlab
Xname=r'C:\Users\sabatini\Documents\behEvents.mat'
yname=r'C:\Users\sabatini\Documents\neuron_data_matrix.mat'

# read files
X=scipy.io.loadmat(Xname)
y=scipy.io.loadmat(yname)
X = X['behEvents']
y = y['neuron_data_matrix']

# flip dimensions of X
X = np.transpose(X)
y = np.transpose(y)

# size of X
Xsize = X.shape
ysize = y.shape

# print size of X
print(Xsize)
print(ysize)

# Set up design matrix by shifting the data by various time steps
nshifts = 30
# Iterate through nshifts
for i in range(nshifts):
    # Shift the data by i time steps
    X_shifted = pd.DataFrame(X).shift(i).fillna(0).values
    # Add the shifted data to the design matrix
    if i == 0:
        X_design = X_shifted
    else:
        X_design = np.concatenate((X_design, X_shifted), axis=1)

# Size of X_design
print(X_design.shape)

# Create the model
model = PoissonRegressor()

# Create the cross-validation object
kf = KFold(n_splits=5)

# Which neuron to analyze
which_neuron = 0

# Get which_neuron of y
y = y[:,which_neuron]

# Do a 5-fold cross-validation of X_design and y
# Iterate through the folds
for train_index, test_index in kf.split(X_design):
    # Get the training and test sets
    X_train, X_test = X_design[train_index], X_design[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model on the shifted training data
    model.fit(X_train, y_train)

    # Evaluate the model on the non-shifted test data
    score = model.score(X_test, y_test)

    # Print the evaluation score
    print(score)