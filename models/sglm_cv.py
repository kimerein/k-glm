import pandas as pd
import numpy as np
from models import sglm
import itertools
import threading, queue
from multiprocessing import Process, Pool
import time
from models import split_data

# TODO: Multidimensional Array -- paramgrid and output grid
# TODO: Add an OrderedDict implementation for the generate_mult_params version

# TODO: Add a Feature Selection methodology -- to adjust feature selection based on cross-validation

# Trial-based splitting (remove inter-trial information?)

def simple_cv_fit(X, y, cv_idx, glm_kwarg_lst, model_type='Normal', verbose=0, score_method='mse'):
    """
    Fit the desired model using the list of keyword arguments provided in
    glm_kwarg_lst, identify the best model, and return the associated
    score, parameters, and the model itself.

    JZ 2021
    
    Args:
        X : pd.DataFrame
            Predictor DataFrame to fit
        y : pd.Series
            Response Series to fit
        cv_idx : list(tuple(tuple(int)))
            List of list of indices to use for fold Cross Validation —
            k-folds list [ ( training tuple(indices), testing tuple(indices) ) ]
        glm_kwarg_lst : list(dict)
            List of dictionaries of keyword arguments to try for validation parameter search
        model_type : str
            Keyword arguments to be passed to GLM model
        verbose ; int
            Amount of information to print out during model fitting / validation (larger numbers print more)
        score_method : str
            Either 'mse' or 'r2' to base cross-validation selection on Mean Squared Error or R^2

    Returns: From the model with the best (largest) score value, return the...
             Best Score Value, Best Score Standard Deviation, Best Params, Best Model
    """
    # Step 4: Fit GLM models for all possible sets of values
    cv_results = cv_glm_mult_params(X.values,
                                    y.values,
                                    cv_idx,
                                    model_type,
                                    glm_kwarg_lst,
                                    verbose=verbose,
                                    score_method=score_method
                                    # [tuple([glm_kwarg[_] for _ in []]) for glm_kwarg in glm_kwarg_lst]
                                    )
    best_score = cv_results['best_score']
    best_score_std = cv_results['best_score_std']
    best_params = cv_results['best_params']
    best_model = cv_results['best_model']
    return best_score, best_score_std, best_params, best_model, cv_results



def cv_idx_by_timeframe(X, y=None, timesteps_per_bucket=20, num_folds=10, test_size=None):
    """
    Generate Cross Validation indices by keeping together bucketed timesteps
    (bucketing together timesteps between intervals of timesteps_per_bucket).

    JZ 2021
    
    Args:
        X : pd.DataFrame
            Prediction DataFrame from which to bucket
        y : pd.Series
            Response Series
        timesteps_per_bucket : int
            Number of timesteps (i.e. rows in the DataFrame) that should be kept together as buckets
        num_folds : int
            Number of Cross Validation segmentations that should be used for k-fold Cross Validation
        test_size : float
            Percentage of datapoints to use in each GroupShuffleSplit fold for validation
    
    Returns: List of tuples of indices to be used for validation / hyperparameter selection
    """
    bucket_ids = split_data.bucket_ids_by_timeframe(X.shape[0], timesteps_per_bucket=timesteps_per_bucket)
    cv_idx = split_data.cv_idx_from_bucket_ids(bucket_ids, X, y=y, num_folds=num_folds, test_size=test_size)
    return cv_idx


class SGLM_worker():
    def __init__(self, queue, verbose=0):
        if verbose > 1:
            print('Worker created')
        self.queue = queue
        self.verbose = verbose

    def run_single(self):
        while True:
            if self.verbose > 1:
                print('Running single')
            # if self.queue.empty():
            #     return
            try:
                glm, args, kwargs = self.queue.get(block=True, timeout=3)
            except queue.Empty as qe:
                if self.verbose > 1:
                    print('Single run queue completed')
                return
            glm.fit_set(*args, **kwargs)
            self.queue.task_done()
            # if self.queue.empty():
            #     return
    
    def run_multi(self):
        while True:
            # if self.verbose > 1:
            print('Running multi')
            # if self.queue.empty():
            #     return
            try:
                args, kwargs = self.queue.get(block=True, timeout=3)
            except queue.Empty as qe:
                # if self.verbose > 1:
                print('Multirun queue completed')
                return
            cv_glm_single_params(*args, **kwargs)
            self.queue.task_done()
            # if self.queue.empty():
            #     return

def cv_glm_single_params(X, y, cv_idx, model_name, glm_kwargs, verbose=0, resp_list=[], beta_=None, beta0_=None, score_method='mse'):
    """
    Runs cross-validation on GLM for a single set of parameters.

    JZ 2021
    
    Args:
        X : np.ndarray
            The full set of available predictor data (columns should be features, rows should be timesteps).
        y : np.ndarray
            The full set of corresponding available response data.
        cv_idx : list of pairs of lists
            Cross-validation indices. Each entry in the outer list is
            for a different run. Each entry in the outer list should 
            contain 2 lists, the first one containing the training 
            indices, and the second one containing the test indices.
        model_name : str
            The type of GLM to construct ('Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', or 'Multinomial')
        glm_kwargs : dict
            Keyword arguments to pass to the GLM constructor
        verbose : int
            How much information to print during the fititng process
        resp_list : list
            list for in place appending of fitting results (for use in multi-threading)
        beta_ : np.ndarray
            Pre-initialized coefficient values for warm-start-based fitting
        beta0_ : np.ndarray
            Pre-initialized intercept value for warm-start-based fitting
        score_method : str
            'mse' for Mean Squared Error-based scoring, 'r2' for R^2 based scoring
    
    Returns: dict of information relevant to fitted validation model based on single set of GLM parameters
    """

    
    q = queue.Queue()

    threads = []

    n_coefs = X.shape[1]
    n_idx = len(cv_idx)

    roll = glm_kwargs.pop('roll', 0)
    y_rolled = np.roll(y.reshape(-1), roll)

    cv_coefs = np.zeros((n_coefs, n_idx))
    cv_intercepts = np.zeros(n_idx)
    cv_scores_train = np.zeros(n_idx)
    cv_scores_test = np.zeros(n_idx)

    resids = []
    mean_resids = []

    for iter_cv, (idx_train, idx_test) in enumerate(cv_idx):
        X_train = X[idx_train,:]
        y_train = y_rolled[idx_train]
        X_test = X[idx_test,:]
        y_test = y_rolled[idx_test]

        # if iter_cv == 0 and not isinstance(beta_, np.ndarray) and beta0_ is None:
        #     start = time.time()

        #     pca_glm = sglm.GLM(model_name, **glm_kwargs)
        #     pca_glm.pca_fit(X_train, y_train)
        #     print(f'> PCA GLM Built in {time.time() - start} seconds')

        #     beta0_ = pca_glm.beta0_
        #     beta_ = pca_glm.beta_.copy()
    
        glm = sglm.GLM(model_name, beta0_=beta0_, beta_=beta_, **glm_kwargs, score_method=score_method)
        args = (X_train, y_train, X_test, y_test,
                cv_coefs, cv_intercepts, cv_scores_train, cv_scores_test,
                iter_cv,)
        kwargs = {'resids': resids, 'mean_resids': mean_resids,
                  'id_fit': iter_cv, 'verbose': verbose
                 }

        q.put((glm, args, kwargs))
    
    num_workers = 3
    workers = [SGLM_worker(q) for _ in range(num_workers)]
    threads = [threading.Thread(target=worker.run_single, daemon=True) for worker in workers]
    for thread in threads:
        thread.start()
    
    q.join()
    for thread in threads:
        thread.join()

    #####################
    
#     # L2 1/(2N) Ratio Fix
#     if 'l1_ratio' in glm_kwargs and glm_kwargs['l1_ratio'] == 0.0:
#         glm_kwargs['alpha'] = glm_kwargs['alpha']
        
        
    glm = sglm.GLM(model_name, beta0_=beta0_, beta_=beta_, **glm_kwargs)
    glm.fit(X, y)

    #####################

    if verbose > 0:
        print('Completing arguments:', glm_kwargs)

    ret_dict = {
        'cv_coefs': cv_coefs,
        'cv_intercepts': cv_intercepts,
        'cv_scores_train': cv_scores_train,
        'cv_scores_test': cv_scores_test,
        'cv_mean_score_train': np.mean(cv_scores_train),
        'cv_mean_score': np.mean(cv_scores_test),
        'cv_std_score': np.std(cv_scores_test),
        'cv_R2_score': sglm.calc_R2(np.concatenate(resids), np.concatenate(mean_resids)),
        'cv_mse_score': np.mean(np.square(np.concatenate(resids))),
        'glm_kwargs': glm_kwargs,
        'model': glm
    }

    print(f"{glm_kwargs}\n> cv_mean_score_train: {ret_dict['cv_mean_score_train']}\n> cv_R2_score: {ret_dict['cv_R2_score']}\n> cv_mean_score: {ret_dict['cv_mean_score']}")

    resp_list.append(ret_dict)

    return ret_dict



def cv_glm_mult_params(X, y, cv_idx, model_name, glm_kwarg_lst, verbose=0, score_method='mse'):
    """
    Runs cross-validation on GLM over a list of possible parameters.

    JZ 2021
    
    Args:
        X : np.ndarray
            The full set of available predictor data (columns should be features, rows should be timesteps).
        y : np.ndarray
            The full set of corresponding available response data (1D).
        cv_idx : list of pairs of lists
            Cross-validation indices. Each entry in the outer list is
            for a different run. Each entry in the outer list should 
            contain 2 lists, the first one containing the training 
            indices, and the second one containing the test indices.
        model_name : str
            The type of GLM to construct ('Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', or 'Multinomial')
        glm_kwarg_lst : list(dict(GLM parameters))
            A list of all kwarg parameters for the GLM through which
            the crossvalidation function should iterate. ('roll' should be
            specified here if desired where 'roll' corresponds to the amount
            of 1D index shifts that should be applied)
        verbose : int
            How much information to print during the fititng process
        score_method : str
            'mse' for Mean Squared Error-based scoring, 'r2' for R^2 based scoring
    
    Returns: dict of information relevant to the best model identified and overall fit information
    """
    q2 = queue.Queue()

    final_results = {}
    best_score = -np.inf #if score_metric == 'pseudo_R2' else np.inf
    best_params = None

    threads = []
    resp = list()


    start = time.time()

#     pca_glm = models.sglm.GLM('PCA Normal') if model_name in {'Normal', 'Gaussian'} else models.sglm.GLM(model_name)
#     pca_glm.pca_fit(X, y)
#     print(f'> PCA GLM Built in {time.time() - start} seconds')

    beta0_ = None
    beta_ = None


    for i, glm_kwargs in enumerate(glm_kwarg_lst):
        print(glm_kwargs)

        model_name = glm_kwargs.pop('model_name', 'Gaussian')

        args = (X, y, cv_idx, model_name, glm_kwargs,)
        kwargs = {'verbose': verbose,
                  'resp_list': resp,
                  'beta0_':beta0_,
                  'beta_':beta_,
                  'score_method':score_method
                 }

        q2.put((args, kwargs))
        
    num_workers = 4
    workers = [SGLM_worker(q2) for _ in range(num_workers)]
    threads = [threading.Thread(target=worker.run_multi, daemon=True) for worker in workers]
    for thread in threads:
        thread.start()
    q2.join()
    for thread in threads:
        thread.join()

    if len(resp) == 0:
        print('len resp', len(resp))
        print('resp', resp)
    
    for cv_result in resp:
        if (score_method == 'r2' and cv_result['cv_R2_score'] > best_score):
            best_score = cv_result['cv_R2_score']
            best_score_std = cv_result['cv_std_score']
            best_params = cv_result['glm_kwargs']
            best_model = cv_result['model']
        elif (score_method == 'mse' and cv_result['cv_mean_score'] > best_score):
            best_score = cv_result['cv_mean_score']
            best_score_std = cv_result['cv_std_score']
            best_params = cv_result['glm_kwargs']
            best_model = cv_result['model']
            
    
    final_results = {
        'best_score': best_score,
        'best_score_std': best_score_std,
        'best_params': best_params,
        'best_model': best_model,
        'full_cv_results': resp,
    }

    return final_results



def generate_mult_params(kwarg_lists, kwargs=None):
    """
    Generates a list of dictionaries of all possible parameter combinations
    from a dictionary of lists.

    JZ 2021
    
    Args:
        kwarg_lists : dict(list(keywords))
            Dictionary where each key is associated with a list of possible parameters to consider.
        kwargs : Optional[dict(keywords)]
            Dictionary of fixed keyword arguments that should remain the same across all CV trials.
    
    Returns: list of dicts of keyword arguments for GLM model fitting
    """

    base_list = [[kwargs]] if kwargs else []
    flipped_dict_list = base_list + [[{key:_} for _ in kwarg_lists[key]] for key in kwarg_lists]
    cart_prod = list(itertools.product(*flipped_dict_list))

    return [{_key:dct[_key] for dct in cart_prod[i] for _key in dct} for i in range(len(cart_prod))]


# def execute_backward_selection(X, y, *args, criterion='AIC', **kwargs):

#     """
#     Generates a list of dictionaries of all possible parameter combinations
#     from a dictionary of lists.

#     Parameters
#     ----------
#     X : np.ndarray
#         The full set of available predictor data (columns should be features, rows should be timesteps).
#     y : np.ndarray
#         The full set of corresponding available response data (1D).
#     GLM : sglm.GLM or sglm.SKGLM class
#         GLM model to fit
#     args : Optional[dict(keywords)]
#         Positional arguments to GLM that should remain the same across all CV trials.
#     criterion : str — 'AIC' or 'BIC'
#         The criterion to use for backward selection (note that these are pseudo-AIC / BIC for now since likelihood varies by link function)
#     kwargs : Optional[dict(keywords)]
#         Dictionary of fixed keyword arguments to GLM that should remain the same across all CV trials.
#     """

#     def calc_criterion(X, y, model, criterion):
#         score = model.score(X, y)
#         neg_score = 1 - score
#         params = 





