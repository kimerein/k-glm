import numpy as np
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score

def kim_ovo():
    # Load dataset
    X, y, timepoints, time_step = load_dataset()
    print(X.shape)

    # Train SVM classifier with OvO strategy
    trainOVO(X, y)

    # Make 0 and 1 the same label, and make 2 and 3 the same label
    # This puts successes together, failures together
    # Make a copy of y
    backupy=y.copy()
    y[y == 0] = 1
    y[y == 2] = 3
    # Train this
    trainOVO(X, y)

    # Make 0 and 2 the same label, and make 1 and 3 the same label
    y=backupy.copy()
    y[y == 0] = 2
    y[y == 1] = 3
    print(y)
    # Train this
    trainOVO(X, y)


def trainOVO(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # Create SVM classifier with OvO strategy
    svm = OneVsOneClassifier(SVC(kernel='linear'))

    # Perform 5-fold cross-validation on training set
    scores = cross_val_score(svm, X_train, y_train, cv=5)

    # Print cross-validation scores
    print("Cross-validation scores:", scores)

    # Train SVM classifier on full training set
    svm.fit(X_train, y_train)

    # Evaluate test accuracy on held-out test set
    test_accuracy = svm.score(X_test, y_test)
    print("Test accuracy:", test_accuracy)


def load_dataset():

    import scipy.io

    # define files to import from Matlab
    Xname=r'C:\Users\sabatini\Documents\currtens\tensor.mat'
    yname=r'C:\Users\sabatini\Documents\currtens\allLabels.mat'
    timename=r'C:\Users\sabatini\Documents\currtens\timepoints_for_tensor.mat'
    # read data from files
    X=scipy.io.loadmat(Xname)
    y=scipy.io.loadmat(yname)
    timepoints=scipy.io.loadmat(timename)
    X = X['tensor']
    y = y['allLabels']
    timepoints = timepoints['timepoints_for_tensor']
    timepoints=np.squeeze(timepoints)
    time_step = np.median(np.diff(timepoints, 1, 0))
    print(timepoints)
    print(time_step)
    print(X.shape)
    print(y.shape)

    # Linearize first two dimensions of tensor
    X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
    print(X.shape)

    # Trials should be first dimension and features should be second dimension
    X = X.T
    print(X.shape)
    y = np.squeeze(y)
    print(y.shape)

    # Fill all nans with zeros
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    return X, y, timepoints, time_step

kim_ovo()