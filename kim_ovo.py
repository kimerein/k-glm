import numpy as np
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

def kim_ovo():
    # Load dataset
    X, y, timepoints, time_step = load_dataset()
    print(X.shape)

    # Train SVM classifier with OvO strategy
    # Train OVO but only get the test accuracy
    testaccuracy, shuffleaccuracy = trainOVOloop(X, y, whichKernel='linear', dispOutput=False)
    title = 'Distribution of Test and Shuffle Accuracies for Linear Kernel'
    vioPlots(testaccuracy, shuffleaccuracy, title)
    #testaccuracy, shuffleaccuracy = trainOVOloop(X, y, whichKernel='poly', dispOutput=False)
    #vioPlots(testaccuracy, shuffleaccuracy, 'Distribution of Test and Shuffle Accuracies for Polynomial Kernel')
    #testaccuracy, shuffleaccuracy = trainOVOloop(X, y, whichKernel='rbf', dispOutput=False)
    #vioPlots(testaccuracy, shuffleaccuracy, 'Distribution of Test and Shuffle Accuracies for RBF Kernel')
    #testaccuracy, shuffleaccuracy = trainOVOloop(X, y, whichKernel='sigmoid', dispOutput=False)
    #vioPlots(testaccuracy, shuffleaccuracy, 'Distribution of Test and Shuffle Accuracies for Sigmoid Kernel')
    # If train on shuffled y
    #testaccuracy, shuffleaccuracy = trainOVOloop(X, y = np.random.permutation(y), whichKernel='linear', dispOutput=False)
    #title = 'SHUFFLE TRAIN Distribution of Test and Shuffle Accuracies for Linear Kernel'
    #vioPlots(testaccuracy, shuffleaccuracy, title)
    print()

    # Make 0 and 1 the same label, and make 2 and 3 the same label
    # This puts successes together, failures together
    # Make a copy of y
    backupy=y.copy()
    y[y == 0] = 1
    y[y == 2] = 3
    # Train this
    testaccuracy, shuffleaccuracy = trainOVOloop(X, y, whichKernel='linear', dispOutput=False)
    title = 'Distribution of Test and Shuffle Accuracies for Linear Kernel'
    vioPlots(testaccuracy, shuffleaccuracy, title)
    #testaccuracy, shuffleaccuracy = trainOVOloop(X, y, whichKernel='poly', dispOutput=False)
    #vioPlots(testaccuracy, shuffleaccuracy, 'Distribution of Test and Shuffle Accuracies for Polynomial Kernel')
    #testaccuracy, shuffleaccuracy = trainOVOloop(X, y, whichKernel='rbf', dispOutput=False)
    #vioPlots(testaccuracy, shuffleaccuracy, 'Distribution of Test and Shuffle Accuracies for RBF Kernel')
    #testaccuracy, shuffleaccuracy = trainOVOloop(X, y, whichKernel='sigmoid', dispOutput=False)
    #vioPlots(testaccuracy, shuffleaccuracy, 'Distribution of Test and Shuffle Accuracies for Sigmoid Kernel')
    print()

    # Make 0 and 2 the same label, and make 1 and 3 the same label
    y=backupy.copy()
    y[y == 0] = 2
    y[y == 1] = 3
    # Train this
    testaccuracy, shuffleaccuracy = trainOVOloop(X, y, whichKernel='linear', dispOutput=False)
    title = 'Distribution of Test and Shuffle Accuracies for Linear Kernel'
    vioPlots(testaccuracy, shuffleaccuracy, title)
    #testaccuracy, shuffleaccuracy = trainOVOloop(X, y, whichKernel='poly', dispOutput=False)
    #vioPlots(testaccuracy, shuffleaccuracy, 'Distribution of Test and Shuffle Accuracies for Polynomial Kernel')
    #testaccuracy, shuffleaccuracy = trainOVOloop(X, y, whichKernel='rbf', dispOutput=False)
    #vioPlots(testaccuracy, shuffleaccuracy, 'Distribution of Test and Shuffle Accuracies for RBF Kernel')
    #testaccuracy, shuffleaccuracy = trainOVOloop(X, y, whichKernel='sigmoid', dispOutput=False)
    #vioPlots(testaccuracy, shuffleaccuracy, 'Distribution of Test and Shuffle Accuracies for Sigmoid Kernel')
    print()


def vioPlots(testaccuracy_vec, shuffleaccuracy_vec, title):
    import matplotlib.pyplot as plt
    from scipy.stats import shapiro, ttest_ind, mannwhitneyu
    import numpy as np

    # test for normality
    _, p1 = shapiro(testaccuracy_vec)
    _, p2 = shapiro(shuffleaccuracy_vec)

    if p1 > 0.05 and p2 > 0.05:
        # if both vectors are normally distributed, perform a t-test
        t, p = ttest_ind(testaccuracy_vec, shuffleaccuracy_vec)
        if p < 0.05:
            print('Means are significantly different')
            print('p = ', p)
        else:
            print('Means are not significantly different')
            print('p = ', p)
    else:
        # if not normally distributed, perform a Mann-Whitney U test
        u, p = mannwhitneyu(testaccuracy_vec, shuffleaccuracy_vec)
        if p < 0.05:
            print('Distributions are significantly different')
            print('p = ', p)
        else:
            print('Distributions are not significantly different')
            print('p = ', p)

    # Create a figure with a single subplot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Create a violin plot of the test accuracy distribution
    ax.violinplot(testaccuracy_vec, positions=[1], showmeans=True, showextrema=True)

    # Create a violin plot of the shuffle accuracy distribution
    ax.violinplot(shuffleaccuracy_vec, positions=[2], showmeans=True, showextrema=True)

    # Add labels and title
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Test Accuracy', 'Shuffle Accuracy'])
    ax.set_ylabel('Accuracy')
    ax.set_title(title)

    plt.show()


def trainOVOloop(X, y, whichKernel, dispOutput):
    n = 100  # number of iterations
    testaccuracy_vec = []  # vector to store test accuracies
    shuffleaccuracy_vec = []  # vector to store shuffle accuracies

    for i in range(n):
        _, testaccuracy, shuffleaccuracy = trainOVO(X, y, whichKernel='linear', dispOutput=False)
        testaccuracy_vec.append(testaccuracy)
        shuffleaccuracy_vec.append(shuffleaccuracy)
    
    return testaccuracy_vec, shuffleaccuracy_vec


def trainOVO(X, y, whichKernel, dispOutput):

    import sklearn.utils.class_weight

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #print(X_train.shape)
    #print(X_test.shape)
    #print(y_train.shape)
    #print(y_test.shape)

    # Compute class weights
    class_w = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    # Make class_w a dictionary
    class_w = dict(enumerate(class_w))

    # Create SVM classifier with OvO strategy
    svm = OneVsOneClassifier(SVC(kernel=whichKernel, class_weight=class_w))

    # Perform 5-fold cross-validation on training set
    #scores = cross_val_score(svm, X_train, y_train, cv=5)
    scores = cross_val_score(svm, X_train, y_train, cv=2)

    if dispOutput:
        # Print cross-validation scores
        print("Cross-validation scores:", scores)

    # Train SVM classifier on full training set
    svm.fit(X_train, y_train)

    # Evaluate test accuracy on held-out test set
    test_accuracy = svm.score(X_test, y_test)
    if dispOutput:
        print("Test accuracy for kernel ", whichKernel, ': ', test_accuracy)

    # Evaluate test accuracy on label shuffle of held-out test set
    y_test_shuffled = np.random.permutation(y_test)
    test_accuracy_shuffled = svm.score(X_test, y_test_shuffled)
    if dispOutput:
        print("Test accuracy on shuffled labels: ", test_accuracy_shuffled)

    return svm, test_accuracy, test_accuracy_shuffled


def simulateMoreCells(X, y, n_cells, n_trials):
    import numpy as np
    
    k = 0
    moreCells_subX = np.zeros((n_cells, X.shape[1], n_trials * np.unique(y).shape[0]))
    moreCells_suby = np.zeros((n_trials * np.unique(y).shape[0]))
    for i in np.unique(y):
        # Get indices of y where y == i
        idx = np.where(y == i)[0]
        print(idx)
        subX = X[:, :, idx].copy()
        print(subX.shape)
        for j in range(n_trials):
            moreCells_subX[:,:,k] = randomlySampleCells(subX, n_cells)
            moreCells_suby[k] = i
            k = k+1
      
    return moreCells_subX, moreCells_suby


def randomlySampleCells(X, n_rows):
    import numpy as np

    # randomly sample n rows from 3D matrix
    samples = np.zeros((n_rows, X.shape[1]))

    for i in range(n_rows):
        # randomly choose index from the third dimension
        rand_idx = np.random.randint(X.shape[2])
        
        # randomly choose index from the first dimension
        rand_row_idx = np.random.randint(X.shape[0])
        
        # concatenate selected row to samples
        samples[i] = X[rand_row_idx, :, rand_idx]
    
    return samples


def averageTrialBatches(x, y, nTrialsToAverage):
    import numpy as np

    # average sets of nTrialsToAverage neighboring elements along dimension 3
    x_avg = np.mean(x[:, :, :x.shape[2]//nTrialsToAverage*nTrialsToAverage].reshape(x.shape[0], x.shape[1], x.shape[2]//nTrialsToAverage, nTrialsToAverage), axis=3)

    # do same average along dimension 0 for y
    y_avg = np.mean(y[:y.shape[0]//nTrialsToAverage*nTrialsToAverage].reshape(y.shape[0]//nTrialsToAverage, nTrialsToAverage), axis=1)

    # discard the last few elements that cannot be grouped into sets of nTrialsToAverage
    #x_avg = x_avg[:, :, :x.shape[2]//nTrialsToAverage*nTrialsToAverage]

    return x_avg, y_avg


def load_dataset():

    import scipy.io
    from scipy.ndimage import gaussian_filter1d

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

    # SHUFFLING DATA
    #y = np.random.permutation(y)

    # Smooth each trial with a Gaussian kernel
    #X = gaussian_filter1d(X, sigma=3, axis=1, mode='nearest')

    # Simulate more cells
    n_cells = 50
    n_trials = 50 # per trial condition
    X, y = simulateMoreCells(X, y, n_cells, n_trials)
    print(X.shape)
    print(y.shape)

    # Average across batches of trials
    # nTrialsToAverage = 3
    # # Only average trials that are the same trial type
    # avX = []
    # avy = []
    # for i in np.unique(y):
    #     idx = np.where(y == i)[0]
    #     newX, newy = averageTrialBatches(X[:, :, idx], y[idx], nTrialsToAverage)
    #     # concatenate to avX and avy along 3rd dimension
    #     if i == np.unique(y)[0]:
    #         avX = newX
    #         avy = newy
    #     else:
    #         avX = np.concatenate((avX, newX), axis=2)
    #         avy = np.concatenate((avy, newy), axis=0)
    # X = avX
    # y = avy
    # print(X.shape)
    # print(y.shape)

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

    # Exclude trials with all zeros
    y=y[np.any(X>0.5, axis=1)]
    X = X[np.any(X>0.5, axis=1), :]

    return X, y, timepoints, time_step

kim_ovo()