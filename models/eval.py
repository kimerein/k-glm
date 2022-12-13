import numpy as np
import time
from sglm.models import sglm

def calc_l1(coeffs):
    return np.sum(np.abs(coeffs))

def calc_l2(coeffs):
    return np.sum(np.square(coeffs))


def print_best_model_info(X_setup, best_score, best_params, best_model, start, show_non_zero_coefs=False):
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

    if show_non_zero_coefs:
        # Print out all non-zero coefficients
        print('Non-Zero Coeffs:')
        epsilon = 1e-10
        for ic, coef in enumerate(best_model.coef_):
            if np.abs(coef) > epsilon:
                print(f'> {coef}: {X_setup.columns[ic]}')

    # Print out information related to the best model
    print(f'Best Score: {best_score}')
    print(f'Best Params: {best_params}')
    print(f'Best Model: {best_model}')
    print(f'Best Model — Intercept: {best_model.intercept_}')

    # Print out runtime information
    print(f'Overall RunTime: {time.time() - start}')

    print()
    return


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



