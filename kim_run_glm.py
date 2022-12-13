
import os
import sys
dir_path = '/Users/sabatini/GitHub/'
sys.path.append(f'{dir_path}/k-glm/models')
sys.path.append(f'{dir_path}/k-glm/models')
import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import threading, queue
from multiprocessing import Process, Pool
import time
from models import sglm
from models import sglm_cv
from models import split_data

def kim_run_glm():

    # define files to import from Matlab
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

    print("kim_run_glm beginning")
 
    folds = 5  # k folds for cross validation
    pholdout = 0.2 
    pgss = 0.2      
    score_method = 'mse' 
    glm_hyperparams = [{
        'alpha': 0.0, # 0 is OLS
        'l1_ratio': 0.0,
        'max_iter': 1000,
        'fit_intercept': False
    }]

    # Keep track of time
    start = time.time()

    # Timeshifts
    # Set up design matrix by shifting the data by various time steps
    a=-30
    b=30
    nshifts = list(range(a, b+1))
    print(nshifts)

    # Iterate through nshifts when nshifts is a list
    for i in nshifts:        
        # Shift the data by i time steps
        # Exclude last column of X from design matrix, because this is trial number
        X_shifted = pd.DataFrame(X).iloc[:, :-1].shift(i).fillna(0).values
        # Add the shifted data to the design matrix
        # If i is the first element of nshifts, then X_design is X_shifted
        # Otherwise, concatenate X_design and X_shifted
        if i == nshifts[0]:
            X_design = X_shifted
        else:
            X_design = np.concatenate((X_design, X_shifted), axis=1)

    # Add in neuron data to X_design
    X_design = np.concatenate((X_design, y), axis=1)

    # Add back nTrials to X_design
    X_design = np.concatenate((X_design, X[:, -1].reshape(-1, 1)), axis=1)

    # Make a heatmap plot of X_design
    plt.figure(figsize=(20, 10))
    # Subset of X_design
    X_design_sub = X_design[0:100, 0:100]
    # Get number of columns of y
    # Add last n trials of X_design to X_design_sub
    X_design_sub = np.concatenate((X_design_sub, X_design[0:100, -(ysize[1]+1):]), axis=1)
    sns.heatmap(X_design_sub, cmap='viridis')
    plt.show()
    print(X_design.shape)

    # Fix for interp error from Matlab
    X_design[X_design < 0.000001] = 0

    # Name last column of X_design 'nTrials'
    # Name neuron columns of X_design 'neuron0', 'neuron1', etc.
    X_design = pd.DataFrame(X_design)
    for i in range(X_design.shape[1]):
        if i == X_design.shape[1]-1:
            X_design.rename(columns={i: 'nTrial'}, inplace=True)
        else: 
            if i >= X_design.shape[1]-ysize[1]-1:
                X_design.rename(columns={i: f'neuron{i-(X_design.shape[1]-ysize[1]-1)}'}, inplace=True)
            else:
                X_design.rename(columns={i: f'event{i}'}, inplace=True)
    print(X_design.head())

    # pause code
    input("Press Enter to continue...")

    res = {} # results dictionary

    whichneuron = 'neuron0'

    # Split data into setup (training) and holdout (test) sets
    np.random.seed(30186)

    dfrel_setup, dfrel_holdout, holdout_mask = split_data.holdout_splits(X_design,
                                                id_cols=['nTrial'],
                                                perc_holdout=pholdout)

    # Get columns of dfrel_setup with 'event' or 'nTrial' in the name
    # These are the columns that will be used for training
    # Exclude columns with 'neuron' in the name
    # These are the columns that will be used for testing
    X_setup_cols = [col for col in dfrel_setup.columns if 'event' in col or 'nTrial' in col]
    X_neuron_cols = [col for col in dfrel_holdout.columns if 'neuron' in col]
    # Use X_setup_cols to subset dfrel_setup and dfrel_holdout
    X_setup, X_holdout = dfrel_setup[X_setup_cols].copy(), dfrel_holdout[X_setup_cols].copy()
    y_setup, y_holdout = dfrel_setup[whichneuron].copy(),  dfrel_holdout[whichneuron].copy()

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




