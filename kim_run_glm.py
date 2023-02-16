
import os
import sys
dir_path = '/Users/sabatini/GitHub/'
sys.path.append(f'{dir_path}/k-glm/models')
sys.path.append(f'{dir_path}/k-glm/models')
import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import itertools
import threading, queue
from multiprocessing import Process, Pool
import time
from models import sglm
from models import sglm_cv
from models import split_data
from visualization import visualize
import os

def kim_run_glm():
    
    # Read csv
    df = pd.read_csv(r'C:\Users\sabatini\Downloads\Spike sorting analysis - Combined phys and photo.csv')
    direcname = df.iloc[:, 7]
    currexpts = range(287, 297)
    currexpts = range(currexpts[0] - 1, currexpts[-1])
    for i in currexpts:
        currdirecname = direcname[i]
        print(currdirecname)
        if df.iloc[i,12]==1:
            continue
        # If directory does not exist, skip
        if not os.path.exists(currdirecname):
            continue
        # Add 'forglm' to directory name
        currdirecname = os.path.join(currdirecname, 'forglm')
        kim_glm(currdirecname, doShuffle=False, suppressPlots=True)


def kim_glm(direcname, doShuffle=False, suppressPlots=True):
    #matplotlib.use('TkAgg')
    matplotlib.rcParams['path.simplify_threshold'] = 0.7

    # define files to import from Matlab
    #doShuffle=False
    #direcname=r'Z:\MICROSCOPE\Kim\WHISPER recs\dLight4\20210120\SU aligned to behavior\forglm'
    #suppressPlots=True
    # Combine directory name and file name
    Xname=os.path.join(direcname, 'behEvents.mat')
    yname=os.path.join(direcname, 'neuron_data_matrix.mat')
    timename=os.path.join(direcname, 'timepoints.mat')
    saveDir=os.path.join(direcname, 'output')
    # If saveDir does not exist, create it
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    #Xname=r'Z:\MICROSCOPE\Kim\WHISPER recs\87\20201224\SU aligned to behavior\forglm\behEvents.mat'
    #yname=r'Z:\MICROSCOPE\Kim\WHISPER recs\87\20201224\SU aligned to behavior\forglm\neuron_data_matrix.mat'
    #timename=r'Z:\MICROSCOPE\Kim\WHISPER recs\87\20201224\SU aligned to behavior\forglm\timepoints.mat'
    #saveDir=r'Z:\MICROSCOPE\Kim\WHISPER recs\87\20201224\SU aligned to behavior\forglm\output'

    # read files
    X=scipy.io.loadmat(Xname)
    y=scipy.io.loadmat(yname)
    timepoints=scipy.io.loadmat(timename)
    X = X['behEvents']
    y = y['neuron_data_matrix']
    timepoints = timepoints['timepoints']

    # Kim's setup of beh events
    # list of event types
    # event_types = ['cue', 'opto', 'distract', 'reachzone', 'fidget', 'success', 'drop', 'missing', 'failure', 'chew']
    event_types = ['cue', 'opto', 'distract', 'success', 'drop', 'miss']

    # flip dimensions of X
    X = np.transpose(X)
    y = np.transpose(y)
    timepoints = np.transpose(timepoints)
    time_step = np.median(np.diff(timepoints, 1, 0))

    # if doShuffle is True, randomly permute trial numbers and randomly circshift behEvents
    if doShuffle:
        # Randomly permute trial numbers
        # Get number of trials
        nTrials = np.max(X[:, -1])
        # Randomly permute trial numbers
        newTrials = np.random.permutation(nTrials)
        # Replace trial numbers in X with newTrials
        X[:, -1] = newTrials[X[:, -1].astype(int) - 1]
        # Randomly circshift behEvents
        # Get number of columns in X, excluding trial number
        nCols = X.shape[1] - 1
        # Randomly circshift each column of X
        for i in range(nCols):
            # Get column of X
            col = X[:, i]
            # Randomly circshift column
            X[:, i] = np.roll(col, np.random.randint(0, len(col)))

    # size of X
    Xsize = X.shape
    ysize = y.shape
    tsize = timepoints.shape

    # print size of X
    print(Xsize)
    print(ysize)
    print(tsize)

    # For each column of X and y, calculate the moving average
    # and downsample by a factor of bin
    bin=1 # was already downsampled in Matlab?
    if bin != 1:
        X, y = downsample_before_design_matrix(X, y, bin)
        # downsample timepoints by bin
        timepoints = timepoints[::bin]

    # For each column of X except the last column, take derivative of X
    # i.e., beginning of each event
    # X up to excluding last column
    # X = just_beginning_of_behavior_events(X)
 
    folds = 5  # k folds for cross validation
    pholdout = 0.1  # proportion of data to hold out for testing 
    pgss = None  # proportion of data to use for generalized cross validation  
    # pgss = 0.1  # proportion of data to use for generalized cross validation     
    score_method = 'r2' # 'mse' or 'r2'
    # Alpha = 0 : OLS
    # Alpha != 0 & l1 ratio = 0 → ridge
    # Alpha != 0 & l1 ratio != 0 → lasso
    # If using ElasticNet
    # Minimizes the objective function:
    # 1 / (2 * n_samples) * ||y - Xw||^2_2
    # + alpha * l1_ratio * ||w||_1
    # + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
    glm_hyperparams = chooseGLMhyperparams()

    # Keep track of time
    start = time.time()

    # Timeshifts
    # Set up design matrix by shifting the data by various time steps
    a=-9
    b=30
    nshifts = list(range(a, b+1))
    print(nshifts)

    # Iterate through nshifts when nshifts is a list
    for shi in nshifts:        
        # Shift the data by i time steps
        # Exclude last column of X from design matrix, because this is trial number
        X_shifted = pd.DataFrame(X).iloc[:, :-1].shift(shi).fillna(0).values
        # Add the shifted data to the design matrix
        # If i is the first element of nshifts, then X_design is X_shifted
        # Otherwise, concatenate X_design and X_shifted
        if shi == nshifts[0]:
            X_design = X_shifted
        else:
            X_design = np.concatenate((X_design, X_shifted), axis=1)

    print(X_design.shape)

    for i in range(0, len(event_types)):
        temp = range(i, X_design.shape[1], len(event_types))
        if i==0:
            reordering = temp
        else:
            reordering = np.concatenate((reordering, temp), axis=0)
    
    print(reordering)
    X_design[:, :] = X_design[:, reordering]

    # Add in neuron data to X_design
    X_design = np.concatenate((X_design, y), axis=1)

    # Add back nTrials to X_design
    X_design = np.concatenate((X_design, X[:, -1].reshape(-1, 1)), axis=1)

    # Make a heatmap plot of X_design
    # Subset of X_design
    X_design_sub = X_design[0:100, 0:100]
    # Get number of columns of y
    # Add last n trials of X_design to X_design_sub
    X_design_sub = np.concatenate((X_design_sub, X_design[0:100, -(ysize[1]+1):]), axis=1)
    if suppressPlots == False:
        sns.heatmap(X_design_sub, cmap='viridis')
        # show
        plt.show()
    print(X_design.shape)

    # Fix for interp error from Matlab
    X_design[X_design < 0.000001] = 0

    # pause code
    #input("Press Enter to continue...")


    # Name last column of X_design 'nTrials'
    # Name neuron columns of X_design 'neuron0', 'neuron1', etc.
    X_design = pd.DataFrame(X_design)
    whichevent = 0
    counterforshifts = 0
    for i in range(X_design.shape[1]):
        if i == X_design.shape[1]-1:
            X_design.rename(columns={i: 'nTrial'}, inplace=True)
        else: 
            if i >= X_design.shape[1]-ysize[1]-1:
                X_design.rename(columns={i: f'neuron{i-(X_design.shape[1]-ysize[1]-1)}'}, inplace=True)
            else:
                # get event type whichevent index into event_types
                event_type = event_types[whichevent]
                print(event_type)
                X_design.rename(columns={i: f'{event_type}_{nshifts[counterforshifts]}'}, inplace=True)
                counterforshifts += 1
                if counterforshifts == len(nshifts):
                    counterforshifts = 0
                    whichevent += 1
    print(X_design.head())

    # pause code
    # input("Press Enter to continue...")

    res = {} # results dictionary

    # Split data into setup (training) and holdout (test) sets
    np.random.seed(30186)

    dfrel_setup, dfrel_holdout, holdout_mask = split_data.holdout_splits(X_design,
                                                id_cols=['nTrial'],
                                                perc_holdout=pholdout)

    # Get columns of dfrel_setup with an element of event_types or 'nTrial' in the name
    # These are the columns that will be used for training
    # Exclude columns with 'neuron' in the name
    X_setup_cols = [col for col in dfrel_setup.columns if any(ev in col for ev in event_types) or 'nTrial' in col]
    # show names of columns of dfrel_setup that will be used for training
    #print(X_setup_cols)
    X_neuron_cols = [col for col in dfrel_holdout.columns if 'neuron' in col]
    # Use X_setup_cols to subset dfrel_setup and dfrel_holdout
    X_setup, X_holdout = dfrel_setup[X_setup_cols].copy(), dfrel_holdout[X_setup_cols].copy()

    backup_X_setup = X_setup.copy()
    backup_X_holdout = X_holdout.copy()

    # Iterate through neurons

    for i in range(ysize[1]):
    #for i in [1]:
        #try:
        # Name of neuron
        whichneuron = f'neuron{i}'

        X_setup = backup_X_setup.copy()
        X_holdout = backup_X_holdout.copy()

        y_setup, y_holdout = dfrel_setup[whichneuron].copy(),  dfrel_holdout[whichneuron].copy()

        # Show size of X_holdout
        # print(X_holdout.shape)
        # Show size of X_setup
        # print(X_setup.shape)

        # Josh's code -- indices for each fold of cross validation
        kfold_cv_idx = split_data.cv_idx_by_trial_id(X_setup,
                                                y=y_setup, 
                                                trial_id_columns=['nTrial'],
                                                num_folds=folds, 
                                                test_size=pgss)

        X_design['holdout_mask'] = holdout_mask  # to reproduce splits after
        
        # Drop nTrial column from X_setup. (It is only used for group 
        # identification in group/shuffle/split)
        X_setup = X_setup.drop(columns=['nTrial']) 
        X_holdout = X_holdout.drop(columns=['nTrial']) 

        # Run GLM
        best_score, _, best_params, best_model, _ = sglm_cv.simple_cv_fit(X_setup, y_setup, kfold_cv_idx, glm_hyperparams, model_type='Normal', verbose=1, score_method=score_method)

        # Print out best model info
        print_best_model_info(X_setup, best_score, best_params, best_model, start)

        # Fit model on training data, and score on holdout data
        glm, holdout_score, holdout_neg_mse_score = training_fit_holdout_score(X_setup, y_setup, X_holdout, y_holdout, best_params)

        # Reconstruction and plot results
        if suppressPlots == False:
            coef_thresh=0.05
            kim_plot_glm_results(timepoints, X_holdout, y_holdout, time_step, i, glm, X_setup, y_setup, X_setup_cols, nshifts, event_types, coef_thresh)
            plt.show()

        # input("Press Enter to continue...")

        # Collect results
        # Get time and date string
        run_id = pd.to_datetime('today')
        print("Run ID:", run_id)
        res[f'{run_id}'] = {'holdout_score':holdout_score,
                        'holdout_neg_mse_score':holdout_neg_mse_score,
                        'best_score':best_score,
                        'best_params':best_params}

        # Save model metadata
        model_metadata = pd.DataFrame({'score_train':glm.r2_score(X_setup, y_setup), 
                            'score_gss':best_score, 
                            'score_holdout':glm.r2_score(X_holdout, y_holdout),
                            'hyperparams': [best_params],
                            'gssids': [kfold_cv_idx]})

        # For every file iterated, for every result value, for every model fitted, print the reslts
        print(f'Final Results:')
        for k in res:
            print(f'> {k}') # print key, filename
            for k_ in res[k]:
                if type(res[k][k_]) != list:
                    print(f'>> {k_}: {res[k][k_]}')
                else:
                    lst_str_setup = f'>> {k_}: ['
                    lss_spc = ' '*(len(lst_str_setup)-1)
                    print(lst_str_setup)
                    for v_ in res[k][k_]:
                        print((f'{lss_spc} R^2: {np.round(v_[0], 5)} — MSE: {np.round(v_[1], 5)} —'+
                            f' L1: {np.round(v_[2], 5)} — L2: {np.round(v_[3], 5)} — '+
                            f'Params: {v_[4]}'))
                    print(lss_spc + ']')

        # Save glm for this neuron
        # Create Mat file
        # Make neuron numbering match Matlab indexing
        whichneuron = f'neuron{i+1}'
        if doShuffle:
            matfile = os.path.join(saveDir,f'{whichneuron}_glm_shuffle.mat')
        else:
            matfile = os.path.join(saveDir,f'{whichneuron}_glm.mat')
        # Write to mat file
        scipy.io.savemat(matfile, {'glm_coef': glm.coef_, 'feature_names': X_setup_cols, 'glm_intercept': glm.intercept_})
        # Convert model_metadata to csv
        if doShuffle:
            model_metadata.to_csv(os.path.join(saveDir,f'{whichneuron}_glm_shuffle_metadata.csv'))
        else:
            model_metadata.to_csv(os.path.join(saveDir,f'{whichneuron}_glm_metadata.csv'))
        #except:
        #    print(f'Error on {whichneuron}')
        #    continue

    # plt.show()

def chooseGLMhyperparams():

    glm_hyperparams = [{
        'alpha': 0.0, # 0 is OLS
        'l1_ratio': 0.0,
        'max_iter': 1000,
        'fit_intercept': False
    }, {
        'alpha': 0.01, # 0 is OLS
        'l1_ratio': 0.0,
        'max_iter': 1000,
        'fit_intercept': False
    }, {
        'alpha': 0.1, # 0 is OLS
        'l1_ratio': 0.0,
        'max_iter': 1000,
        'fit_intercept': False
    }, {
        'alpha': 1, # 0 is OLS
        'l1_ratio': 0.0,
        'max_iter': 1000,
        'fit_intercept': False
    }, {
        'alpha': 0.01, # 0 is OLS
        'l1_ratio': 0.1,
        'max_iter': 1000,
        'fit_intercept': False
    }, {
        'alpha': 0.1, # 0 is OLS
        'l1_ratio': 0.1,
        'max_iter': 1000,
        'fit_intercept': False
    }, {
        'alpha': 1, # 0 is OLS
        'l1_ratio': 0.1,
        'max_iter': 1000,
        'fit_intercept': False
    }, {
        'alpha': 0.01, # 0 is OLS
        'l1_ratio': 0.5,
        'max_iter': 1000,
        'fit_intercept': False
    }, {
        'alpha': 0.1, # 0 is OLS
        'l1_ratio': 0.5,
        'max_iter': 1000,
        'fit_intercept': False
    }, {
        'alpha': 1, # 0 is OLS
        'l1_ratio': 0.5,
        'max_iter': 1000,
        'fit_intercept': False
    }, {
        'alpha': 0.01, # 0 is OLS
        'l1_ratio': 0.9,
        'max_iter': 1000,
        'fit_intercept': False
    }, {
        'alpha': 0.1, # 0 is OLS
        'l1_ratio': 0.9,
        'max_iter': 1000,
        'fit_intercept': False
    }, {
        'alpha': 1, # 0 is OLS
        'l1_ratio': 0.9,
        'max_iter': 1000,
        'fit_intercept': False
    }, {
        'alpha': 0.01, # 0 is OLS
        'l1_ratio': 1,
        'max_iter': 1000,
        'fit_intercept': False
    }, {
        'alpha': 0.1, # 0 is OLS
        'l1_ratio': 1,
        'max_iter': 1000,
        'fit_intercept': False
    }, {
        'alpha': 1, # 0 is OLS
        'l1_ratio': 1,
        'max_iter': 1000,
        'fit_intercept': False
    }]  # hyperparameters for glm

    return glm_hyperparams

def kim_plot_glm_results(timepoints, X_holdout, y_holdout, time_step, i, glm, X_setup, y_setup, X_setup_cols, nshifts, event_types, coef_thresh):

    # change color depending on feature name
    # use a different color for each event_types
    colors = sns.color_palette('colorblind', len(event_types))
    # Get feature names
    feature_names = X_setup.columns

    # Reconstruct the held-out data
    y_true, pred = visualize.reconstruct_signal(glm, X_holdout, y_holdout)
    plt.close()
    # Plot results
    # Set style
    sns.set(style='white', palette='colorblind', context='poster')
    # make figure
    plt.figure(figsize=(10, 5))
    # subplot
    plt.subplot(2, 1, 1)
    # Get coefficients
    coefs = glm.coef_
    #addEventLinesToPlot(X_setup_cols, X_holdout, timepoints, event_types, colors, feature_names)
    addLargeBetasAsLinesToPlot(X_setup_cols, coefs, coef_thresh, X_holdout, timepoints, event_types, colors, feature_names)
    plt.plot(timepoints[0:len(pred)], pred / (1/time_step), label='Pred', alpha=0.5)
    #plt.plot(timepoints[0:len(y_true.values)], scipy.ndimage.gaussian_filter1d(y_true.values / (1/time_step), sigma=2), label='True', alpha=0.5)
    plt.plot(timepoints[0:len(y_true.values)], y_true.values / (1/time_step), label='True', alpha=0.5)
    plt.ylabel('Firing Rate (Hz)')
    # title of this subplot
    plt.title(f'Neuron {i} reconstrution of held-out data')
    # Reconstruct the training data
    y_true, pred = visualize.reconstruct_signal(glm, X_setup, y_setup)
    plt.close()
    plt.subplot(2, 1, 2)
    #addEventLinesToPlot(X_setup_cols, X_setup, timepoints, event_types, colors, feature_names)
    addLargeBetasAsLinesToPlot(X_setup_cols, coefs, coef_thresh, X_setup, timepoints, event_types, colors, feature_names)
    plt.plot(timepoints[0:len(pred)], pred / (1/time_step), label='Pred', alpha=0.5)
    #plt.plot(timepoints[0:len(y_true.values)], scipy.ndimage.gaussian_filter1d(y_true.values / (1/time_step), sigma=2), label='True', alpha=0.5)
    plt.plot(timepoints[0:len(y_true.values)], y_true.values / (1/time_step), label='True', alpha=0.5)
    plt.ylabel('Firing Rate (Hz)')
    plt.xlabel('Time (s)')
    plt.legend(loc='upper right')
    plt.title(f'Neuron {i} reconstrution of training data')
    plt.tight_layout()

    # Get coefficients from model
    coef = glm.coef_
    # Get intercept from model
    intercept = glm.intercept_
    # Get number of features
    num_features = len(coef)
    # Plot coefficients
    plt.figure(figsize=(10, 5))
    # for each feature
    for j in range(num_features):
        # get feature name
        feature_name = feature_names[j]
        # get color
        color = colors[event_types.index(feature_name.split('_')[0])]
        # plot feature as a scatter plot
        plt.scatter(j, coef[j], s=50, color=color, linewidths=0.5)
    # plot intercept
    plt.scatter(num_features, intercept, color='black', linewidths=0.25)
    # Make a black horizontal line at coef_thresh
    plt.axhline(y=coef_thresh, color='black', linestyle='--')
    plt.axhline(y=-coef_thresh, color='black', linestyle='--')
    temp=[_ for _ in range(len(X_setup_cols)-1) if _ % len(nshifts) == 0]
    namesofcols = list(X_setup_cols)
    plt.xticks(range(0,num_features+1,len(nshifts)), [namesofcols[_] for _ in temp] + ['intercept'], rotation=45)
    # Make text on x axis smaller
    plt.tick_params(axis='x', labelsize=8)
    # Add y label
    plt.ylabel('Coefficient')
    plt.title(f'Neuron {i} coefficients')

def addLargeBetasAsLinesToPlot(X_setup_cols, coefs, coef_thresh, X_holdout, timepoints, event_types, colors, feature_names):

    # Find behavior events types, i.e., columns of X_setup_cols, with large coefficients
    largeCoefs = [_ for _, s in enumerate(coefs) if abs(coefs[_]) > coef_thresh]
    timepoints = timepoints[0:len(X_holdout)]
    # Plot vertical lines at the times of the events
    for j in range(len(largeCoefs)):
        # Get column of X_holdout corresponding to this event type
        # Get X_holdout column values for this event type
        unshifted_events = X_holdout[:][X_setup_cols[largeCoefs[j]]]
        colo = colors[event_types.index(feature_names[largeCoefs[j]].split('_')[0])]
        # array of zeros the size of timepoints[unshifted_events.values == 1]
        offsets = np.ones(len(timepoints[unshifted_events.values == 1])) * (event_types.index(feature_names[largeCoefs[j]].split('_')[0]) / len(event_types)) * 0.1
        plt.scatter(timepoints[unshifted_events.values == 1], offsets, s=10, color=colo, alpha=0.3, linewidth=0.1)

def addEventLinesToPlot(X_setup_cols, X_holdout, timepoints, event_types, colors, feature_names):

    # Find elements of X_setup_cols with '_0' in them
    zeroShiftAt = [_ for _, s in enumerate(X_setup_cols) if '_0' in s]
    # Plot vertical lines at the times of the events
    for j in range(len(zeroShiftAt)):
        # Get column of X_holdout corresponding to this event type
        # Get X_holdout column values for this event type
        unshifted_events = X_holdout[:][X_setup_cols[zeroShiftAt[j]]]
        colo = colors[event_types.index(feature_names[zeroShiftAt[j]].split('_')[0])]
        # For all unshifted_events, plot a vertical line
        for k in range(len(unshifted_events.values)):
            if unshifted_events.values[k] == 1:
                plt.scatter(timepoints[k], 0, color=colo, alpha=0.3, linewidth=0.1)
                #plt.axvline(timepoints[k], color=colo, alpha=0.3, linewidth=0.5)

def just_beginning_of_behavior_events(X):

    '''
    Returns the start of each behavior event
    Args:
        X: behavior data where rows are timepoints and columns are different types of beh event
            Note last trial must be the trial number
    Returns:
        X: with just the start of each behavior event
    '''

    Xsize = X.shape

    newX = np.zeros(X.shape)
    for i in range(Xsize[1]-1):
        newX[:,i] = np.concatenate((np.diff(X[:,i],1,0), np.zeros(1)),axis=0)
    newX[newX != 1] = 0
    # Add last column of X to newX
    newX[:,-1] = X[:,-1]
    X=newX

    return X

def downsample_before_design_matrix(X, y, bin):

    '''
    Downsample neuron data and behavior events by same amount
    Args:
        X: behavior data where rows are timepoints and columns are different types of beh event
            Note last trial must be the trial number
        y: neural data where rows are timepoints and columns are different neurons
        bin: bin size to downsample by
    Returns:
        X: downsampled behavior data
        y: downsampled neural data
    '''

    Xsize = X.shape
    ysize = y.shape

    for i in range(Xsize[1]): 
        # if i is 0, initialize newX
        if i==0:
            newX = pd.DataFrame(X[:,i]).rolling(bin).mean().values[::bin]
        else:
            # if is last column of X, i.e., trial number, downsample but don't average
            if i==Xsize[1]-1:
                newX = np.concatenate((newX, X[::bin,i].reshape(newX.shape[0],1)), axis=1)
            else:
                newX = np.concatenate((newX, pd.DataFrame(X[:,i]).rolling(bin).mean().values[::bin]), axis=1)
    for i in range(ysize[1]):
        # if i is 0, initialize newY
        if i==0:
            newY = pd.DataFrame(y[:,i]).rolling(bin).mean().values[::bin]
        else:
            newY = np.concatenate((newY, pd.DataFrame(y[:,i]).rolling(bin).mean().values[::bin]), axis=1)
    inx=newX[:,1:-2]>0
    newX[:,1:-2]=inx
    newX[0,-1]=X[0,-1]
    X=newX
    y=newY
    # Replace Nan with 0
    X[np.isnan(X)] = 0
    y[np.isnan(y)] = 0
    
    # Plot last column of X
    plt.plot(newX[:,-1])

    return X, y

def training_fit_holdout_score(X_setup, y_setup, X_holdout, y_holdout, best_params):
    '''
    Fit GLM on training data, and score on holdout data
    Args:
        X_setup: X training data on which to fit model
        y_setup: response column for training data
        X_holdout: X holdout data on which to score model
        y_holdout: response column for holdout data
        best_params: dictionary of best parameters
    Returns:
        glm: Fitted model
        holdout_score: Score on holdout data
        holdout_neg_mse_score: Negative mean squared error on holdout data
    '''
    # Refit the best model on the full setup (training) data
    glm = sglm.fit_GLM(X_setup, y_setup, **best_params)

    # Get the R^2 and MSE scores for the best model on the holdout (test) data
    holdout_score = glm.r2_score(X_holdout, y_holdout)
    holdout_neg_mse_score = glm.neg_mse_score(X_holdout, y_holdout)

    return glm, holdout_score, holdout_neg_mse_score

def print_best_model_info(X_setup, best_score, best_params, best_model, start):
    """
    Print best model info
    Args:
        X_setup: setup prediction dataframe
        best_score: best score
        best_params: best parameters
        best_model: best model
        start: start time
    """

    print()
    print('---')
    print()

    # Print out all non-zero coefficients
    #print('Non-Zero Coeffs:')
    #epsilon = 1e-10
    #for ic, coef in enumerate(best_model.coef_):
    #    if np.abs(coef) > epsilon:
    #        print(f'> {coef}: {X_setup.columns[ic]}')

    # Print out information related to the best model
    print(f'Best Score: {best_score}')
    print(f'Best Params: {best_params}')
    print(f'Best Model: {best_model}')
    print(f'Best Model — Intercept: {best_model.intercept_}')

    # Print out runtime information
    print(f'Overall RunTime: {time.time() - start}')

    print()
    return

kim_run_glm()




