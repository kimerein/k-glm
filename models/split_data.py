import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def holdout_split_by_trial_id(X, y=None, id_cols=['nTrial_filenum', 'iBlock'], strat_col=None, strat_mode=None, perc_holdout=0.2):
# def holdout_split_by_trial_id(X, y=None, id_cols=['nTrial', 'iBlock'], strat_col=None, strat_mode=None, perc_holdout=0.2):
    """
    Create a True/False pd.Series using Group ID columns to identify the holdout data to
    be used via GroupShuffleSplit.

    JZ 2021
    
    Args:
        X : pd.DataFrame
            Prediction DataFrame from which to bucket
        y : pd.Series
            Response Series
        id_cols : list(str)
            Columns to use to identify bucketing identifiers
        strat_col : str
            Column to use to stratify/balance the holdout data
        strat_mode : str
            Mode to use to stratify the holdout data
            in the set of ['balanced_train', 'balanced_test', 'stratify']
                balanced_train: Have an equal split of classes in the training set
                balanced_test: Have an equal split of classes in the test set
                stratify: Stratify the classes such that they are proportionally represented in the training & holdout sets
        perc_holdout : int
            Percentage of group identifiers to holdout as test set
    
    Returns: pd.Series of True values if it should be heldout, False if it should be part of training/validation
    """

    assert len(X) > 0

    for i, idc in enumerate(id_cols):
        srs_x_idc_str = X[idc].apply(str)
        if i == 0:
            bucket_ids = srs_x_idc_str.str.len().apply(str) + ':' + srs_x_idc_str
        else:
            bucket_ids = bucket_ids + '__' + srs_x_idc_str.str.len().apply(str) + ':' + srs_x_idc_str
    bucket_ids = bucket_ids.astype("category").cat.codes

    #print('bucket_ids', bucket_ids)
    #print('bucket_ids max+  ', bucket_ids.max() + 1)
    #print('bucket_ids.astype("category").cat.codes', bucket_ids.astype("category").cat.codesucket_ids)

    # print(X[idc])
    # print(bucket_ids)

    num_bucket_ids = int(bucket_ids.max() + 1)

    if strat_col is not None:

        strat_df = X[[strat_col]].copy()
        strat_df['bucket_id'] = bucket_ids

        strat_groups = strat_df[strat_col].unique()
        distinct_buckets = [pd.Series(strat_df[strat_df[strat_col] == _]['bucket_id'].unique()) for _ in strat_groups]
        bucket_sizes = np.array([len(_) for _ in distinct_buckets])

        min_bucket_size = bucket_sizes.min()
        bucket_totals = bucket_sizes.sum()

        # print(set_sizes)
        train_distinct_buckets = []
        test_distinct_buckets = []

        if strat_mode == 'balanced_train':

            num_balanced_train_selection = int(min_bucket_size * (1 - perc_holdout))
            for bucket in distinct_buckets:
                train_distinct_buckets.append(np.random.choice(bucket, num_balanced_train_selection, replace=False))
                test_distinct_buckets.append(bucket[~bucket.isin(train_distinct_buckets[-1])])
            test_ids = np.concatenate(test_distinct_buckets)
            pass

        elif strat_mode == 'balanced_test':

            num_balanced_test_selection = int(min_bucket_size * perc_holdout)
            for bucket in distinct_buckets:
                test_distinct_buckets.append(np.random.choice(bucket, num_balanced_test_selection, replace=False))
                train_distinct_buckets.append(bucket[~bucket.isin(test_distinct_buckets[-1])])
            test_ids = np.concatenate(test_distinct_buckets)
            pass

        elif strat_mode == 'stratify':
            for bucket in distinct_buckets:
                test_distinct_buckets.append(np.random.choice(bucket, int(len(bucket)*perc_holdout), replace=False))
                train_distinct_buckets.append(bucket[~bucket.isin(test_distinct_buckets[-1])])
            test_ids = np.concatenate(test_distinct_buckets)
            pass

        else:
            raise ValueError(f'Invalid strat_mode: {strat_mode}')
    else:
        num_buckets_for_test = int(num_bucket_ids * perc_holdout)
        test_ids = np.random.choice(num_bucket_ids, size=num_buckets_for_test, replace=False)
    
    holdout = bucket_ids.isin(test_ids)

    return holdout

def holdout_splits(dfrel_setup, id_cols=['nTrial_filenum'], perc_holdout=0.2):
# def holdout_splits(dfrel_setup, id_cols=['nTrial'], perc_holdout=0.2):
    '''
    Create holdout splits
    Args:
        dfrel_setup: full setup dataframe
        id_cols: list of columns to use as trial identifiers
        perc_holdout: percentage of data to holdout
    Returns:
        dfrel_setup: full setup dataframe
        dfrel_holdout: full holdout dataframe
        holdout: true/false series — true if datapoint should be in holdout, false if should be in training
    '''
    # Create holdout splits
    holdout = holdout_split_by_trial_id(dfrel_setup, id_cols=id_cols, perc_holdout=perc_holdout)
    dfrel_holdout = dfrel_setup.loc[holdout]
    dfrel_setup = dfrel_setup.loc[~holdout]
    return dfrel_setup, dfrel_holdout, holdout


def cv_idx_by_trial_id(X, y=None, trial_id_columns=[], num_folds=5, test_size=None):
    """
    Generate Cross Validation indices by keeping together trial id columns
    (bucketing together by trial_id_columns) via GroupShuffleSplit.

    JZ 2021
    
    Args:
        X : pd.DataFrame
            Prediction DataFrame from which to bucket
        y : pd.Series
            Response Series
        trial_id_columns : list(str)
            Columns to use to identify bucketing identifiers
        num_folds : int
            Number of Cross Validation segmentations that should be used for GroupShuffleSplit fold Cross Validation
        test_size : float
            Percentage of datapoints to use in each GroupShuffleSplit fold for validation (Defaults to 1/num_folds if None)

    Returns: List of tuple of indices to be used for validation / hyperparameter selection
    """
    X = pd.DataFrame(X)

    for i, idc in enumerate(trial_id_columns):
        srs_x_idc_str = X[idc].apply(str)
        if i == 0:
            bucket_ids = srs_x_idc_str.str.len().apply(str) + ':' + srs_x_idc_str
        else:
            bucket_ids = bucket_ids + '__' + srs_x_idc_str.str.len().apply(str) + ':' + srs_x_idc_str
    
    bucket_ids = bucket_ids.astype("category").cat.codes
    
    cv_idx = cv_idx_from_bucket_ids(bucket_ids, X, y=y, num_folds=num_folds, test_size=test_size)
    return cv_idx


def cv_idx_from_bucket_ids(bucket_ids, X, y=None, num_folds=None, test_size=None):
    '''
    Generate a GroupShuffleSplit on the provided bucket ids. (i.e. create a list of
    splits to use in different stages of cross-validation).

    Args:
        bucket_ids : np.ndarray
            Array of grouping ids to use in generating the GroupShuffleSplit
        X : pd.DataFrame
            Prediction DataFrame from which to bucket
        y : pd.Series
            Response Series
        num_folds : int
            Number of Cross Validation segmentations that should be used for GroupShuffleSplit fold Cross Validation
        test_size : float
            Percentage of datapoints to use in each GroupShuffleSplit fold for validation (Defaults to 1/num_folds if None)
    '''
    # Step 2: Create index sets for each of the K-folds based on prior groupings
    if num_folds is None:
        # Defaults to Leave One Out Cross-Validation
        num_folds = bucket_ids.max() + 1
    
    if test_size is None:
        # Defaults to the number of validation datapoints equal to a single fold
        test_size = 1/num_folds
    
    splitter = GroupShuffleSplit(n_splits=num_folds, test_size=test_size)
    cv_idx = list(splitter.split(X, y, bucket_ids))
    return cv_idx
