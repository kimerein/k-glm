import pandas as pd

def get_x(df, x_cols, keep_rows=None):
    '''
    Get x values for fitting/scoring
    Args:
        df: dataframe that includes x_cols
        x_cols: list of column names to include in prediction
        keep_rows: boolean array of which rows to keep
    Returns:
        df[x_cols]: dataframe only including prediction columns and keep_rows
    '''
    if type(keep_rows) != type(None):
        df = df[keep_rows]
    return df[x_cols]

def get_y(df, y_col, keep_rows=None):
    '''
    Get y values for fitting/scoring
    Args:
        df: dataframe that includes x_cols
        y_cols: column name to use for response
        keep_rows: boolean array of which rows to keep
    Returns:
        df[y_col]: dataframe only including response column and keep_rows
    '''
    if type(keep_rows) != type(None):
        df = df[keep_rows]
    return df[y_col]

def get_xy_all_noniti(df, prediction_X_cols_sftd, y_col, noniticol='wi_trial_keep'):
    X_witi = get_x(df, prediction_X_cols_sftd, keep_rows=None)
    y_witi = get_y(df, y_col, keep_rows=None)
    X_noiti = get_x(df, prediction_X_cols_sftd, keep_rows=df[noniticol])
    y_noiti = get_y(df, y_col, keep_rows=df[noniticol])
    return X_witi, y_witi, X_noiti, y_noiti

def setup_glmsave(glmsave, prefix, filename, neg_order, pos_order, X_cols_all, folds, pholdout, pgss, gssid=None):
    glmsave.set_uid(prefix)
    glmsave.set_filename(filename)
    glmsave.set_timeshifts(neg_order, pos_order)
    glmsave.set_X_cols(X_cols_all)
    glmsave.set_gss_info(folds, pholdout, pgss, gssid=None)
    return

def generate_Ab_labels(df_t):
    df_t = df_t.copy()
    df_t['prv_choseLeft'] = df_t['choseLeft'].shift(1)
    df_t['prv_choseRight'] = df_t['choseRight'].shift(1)
    df_t['prv_wasRewarded'] = df_t['wasRewarded'].shift(1)

    df_t['label_1Aa'] = df_t['prv_wasRewarded'].astype(bool).fillna(False)
    df_t['label_2AB'] = ((df_t['choseLeft'] == df_t['prv_choseLeft']) & (df_t['choseRight'] == df_t['prv_choseRight'])).astype(bool).fillna(False)
    df_t['label_2Aa'] = df_t['wasRewarded'].astype(bool).fillna(False)

    df_t['label'] = '  '

    df_t.loc[df_t['label_1Aa'], 'label'] = df_t.loc[df_t['label_1Aa'], 'label'].str.slice_replace(0, 1, 'A')
    df_t.loc[~df_t['label_1Aa'], 'label'] = df_t.loc[~df_t['label_1Aa'], 'label'].str.slice_replace(0, 1, 'a')

    df_t.loc[(df_t['label_2AB']&df_t['label_2Aa']), 'label'] = df_t.loc[(df_t['label_2AB']&df_t['label_2Aa']), 'label'].str.slice_replace(1, 2, 'A')
    df_t.loc[(~df_t['label_2AB']&df_t['label_2Aa']), 'label'] = df_t.loc[(~df_t['label_2AB']&df_t['label_2Aa']), 'label'].str.slice_replace(1, 2, 'B')
    df_t.loc[(df_t['label_2AB']&~df_t['label_2Aa']), 'label'] = df_t.loc[(df_t['label_2AB']&~df_t['label_2Aa']), 'label'].str.slice_replace(1, 2, 'a')
    df_t.loc[(~df_t['label_2AB']&~df_t['label_2Aa']), 'label'] = df_t.loc[(~df_t['label_2AB']&~df_t['label_2Aa']), 'label'].str.slice_replace(1, 2, 'b')

    df_t.loc[df_t['prv_wasRewarded'].isna(), 'label'] = np.nan

    return df_t
