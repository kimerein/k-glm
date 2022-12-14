import os
import sys
from pathlib import Path
dir_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
sys.path.append(f'{dir_path}/sabatinilab-glm/backend')
sys.path.append(f'{dir_path}/..')
sys.path.append(f'{dir_path}/backend')
sys.path.append(f'{dir_path}/../backend')
# sys.path.append('./backend')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
import time
import seaborn as sns

from scipy.fft import fft, ifft
from scipy.signal import welch

def get_coef_name_sets(coef_names, sftd_coef_names):    
    coef_cols = {}

    for coef_name in coef_names:
        if coef_name in ['nTrial', 'nEndTrial']:
            continue
        lst = [_ for _ in sftd_coef_names if coef_name in _.split('_')]
        lst = [_ if _ != coef_name else coef_name+'_0' for _ in lst]
        lst = sorted(lst, key=lambda x: int(x.split('_')[-1]))
        # lst = [_.replace('_0', '') for _ in lst]
        coef_cols[coef_name] = lst
    return coef_cols

def get_single_coef_set(names, lookup):
    # return [int(_.split('_')[-1]) for _ in names], [lookup[_.replace('_0', '')] for _ in names]
    return [int(_.split('_')[-1]) for _ in names], [lookup[_] for _ in names]


def reconstruct_signal(glm, X, y_true=-1):
    """
    Plot a graph of the predicted signal overlaid with the truly observed signal

    Args:
        glm : sglm.GLM
            Model with which to predict X
        X : np.ndarray or pd.DataFrame
            Predictors to use for signal reconstruction
        y : np.ndarray or pd.DataFrame
            True signal for comparison

    Returns: Tuple[y_true, y_hat]
    """
    import seaborn as sns
    sns.set(style='white', palette='colorblind', context='poster')

    pred = glm.predict(X)

    plt.figure(figsize=(20,10))
    plt.plot(X.index, pred, label='Predicted Signal', alpha=0.5)
    
    if type(y_true) != int:
        plt.plot(X.index, y_true, label='True Signal', alpha=0.5)
    
    plt.xlabel('Data Point Index')
    plt.ylabel(f'{"True vs. " if type(y_true) != int else ""}Prediction Reconstructed Signals')
    plt.legend()

    return y_true, pred


def plot_single_coef_set(name, timeshifts, coefs, ax=None, y_lims=None, binsize=None, label=None):
    """
    Plot a single set of coefficients
    Args:
        name : str
            Name of single coefficient plot
        timeshifts : List[float]
            List of timeshifts
        coefs : List[float]
            List of coefficients
        ax : matplotlib.axes.Axes
            Axes to plot on
        y_lims : List[float]
            Limits of y-axis
        binsize : int
            Binsize in milliseconds
    """
    # Binsize in Milliseconds
    x_vals = np.array(timeshifts)
    x_label = 'Timeshifts'

    if binsize is not None:
        x_vals *= binsize
        x_label += ' (ms)'

    if label:
        ax.plot(x_vals, coefs, label=label)
    else:
        ax.plot(x_vals, coefs)

    ax.set_ylim(y_lims)
    ax.set_title(name)

    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel('Coefficient Value', fontsize=15)
    ax.grid(True)
    return

def plot_all_beta_coefs(coeffs, coef_names, sftd_coef_names, plot_width=4, y_lims=None, filename='', plot_name='', binsize=None, fig=None, axs=None, label=None, plot_rows=6):
    """
    Plot all beta coefficients for a given model
    Args:
        glm : sglm.GLM
            Model with which to predict X
        coef_names : List[str]
            List of names of beta coefficients to plot
        sftd_coef_names : List[str]
            List of names of sftd coefficients to plot
        plot_width : int
            Width of plot in inches
        y_lims : List[float]
            Limits of y-axis
        filename : str
            Name of file to save plot to
        plot_name : str
            Name of plot
        binsize : int
            Binsize in milliseconds
        fig : matplotlib.figure.Figure
            Figure to plot on
        axs : List[matplotlib.axes.Axes]
            Axes to plot on
    Returns:
        fig : matplotlib.figure.Figure
            Figure containing plot
        axs : List[matplotlib.axes.Axes]
            Axes containing plots
    """
    if y_lims is None:
        y_lims = (coeffs.min(), coeffs.max())
    # print(y_lims)

    # print('len(sftd_coef_names)', len(sftd_coef_names))

    coef_lookup = {sftd_coef_names[i]:coeffs[i] for i in range(len(sftd_coef_names))}
    coef_cols = get_coef_name_sets(coef_names, sftd_coef_names)
    
    if fig is None or axs is None:
        num_rows = max(len(coef_cols)//plot_width + (len(coef_cols)%plot_width > 0)*1, plot_rows)

        fig, axs = plt.subplots(num_rows, plot_width)
        fig.set_figheight(20)
        fig.set_figwidth(20)

        addl_plot_name = plot_name if plot_name else 'Feature Coefficients by Timeshift'
        fig.suptitle(f'{addl_plot_name}', fontsize=20)

    for icn, coef_name in enumerate(coef_cols):
        # print(icn)
        timeshifts, coefs = get_single_coef_set(coef_cols[coef_name], coef_lookup)
        
        if len(axs.shape) > 1:
            axs_a = axs[icn//plot_width]
        else:
            axs_a = axs
        
        axs_tmp = axs_a[icn%plot_width]

        plot_single_coef_set(coef_name, timeshifts, coefs, axs_tmp, y_lims, binsize=binsize, label=label)
    


    fig.patch.set_facecolor('white')
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename)

    return fig, axs


def plot_power_spectra(y_true_full, y_hat_full):
    """
    Plot the power spectra of the true and predicted signals
    Args:
        y_true_full : np.ndarray
            True signal
        y_hat_full : np.ndarray
            Predicted signal
    """

    fft_ytrue = np.abs(fft(y_true_full.values))
    fft_yhat = np.abs(fft(y_hat_full))

    f_true, Pxx_den_true = welch(y_true_full.values, fs=20)
    f_hat, Pxx_den_hat = welch(y_hat_full, fs=20)
    f_resid, Pxx_den_resid = welch(y_true_full - y_hat_full, fs=20)

    y_lim_min = np.min([*Pxx_den_true, *Pxx_den_hat, *Pxx_den_resid])
    y_lim_max = np.max([*Pxx_den_true, *Pxx_den_hat, *Pxx_den_resid])
    y_lims = (y_lim_min, y_lim_max)

    x_lim_min = np.min([*f_true, *f_hat, *f_resid])
    x_lim_max = np.max([*f_true, *f_hat, *f_resid])
    x_lims = (x_lim_min, x_lim_max)

    fig, ax = plt.subplots(1,3)
    fig.suptitle('Welch Power Spectra of Response / Reconstruction — All Data, Excluding ITI')
    fig.set_figheight(20)
    fig.set_figwidth(40)


    ax[0].semilogy(f_true, Pxx_den_true)
    # plt.ylim([0.5e-3, 1])
    ax[0].set_xlabel('frequency [Hz]')
    ax[0].set_ylabel('PSD [V**2/Hz]')
    ax[0].set_ylim(y_lims)
    ax[0].set_title('Power Spectra of Raw Response')
    ax[0].grid()

    ax[1].semilogy(f_hat, Pxx_den_hat)
    ax[1].set_xlabel('frequency [Hz]')
    ax[1].set_ylabel('PSD [V**2/Hz]')
    ax[1].set_ylim(y_lims)
    ax[1].set_title('Power Spectra of Reconstructed Response')
    ax[1].grid()


    ax[2].semilogy(f_resid, Pxx_den_resid)
    ax[2].set_xlabel('frequency [Hz]')
    ax[2].set_ylabel('PSD [V**2/Hz]')
    ax[2].set_ylim(y_lims)
    ax[2].set_title('Power Spectra of (Raw - Reconstructed)')
    ax[2].grid()

    plt.show()

    fig.patch.set_facecolor('white')
    fig.savefig('figure_outputs/spectral_out.png')

    return

def plot_single_avg_reconstruction(ci_setup_true, ci_setup_pred, ax, min_time, max_time, min_signal, max_signal, x_label, y_label, title):
    """
    Plot the average reconstruction of a single CI setup
    Args:
        ci_setup_true : pd.DataFrame
            CI setup with true signal
        ci_setup_pred : pd.DataFrame
            CI setup with predicted signal
        ax : matplotlib.axes.Axes
            Axes to plot on
        min_time : float
            Minimum time to plot
        max_time : float
            Maximum time to plot
        min_signal : float
            Minimum signal to plot
        max_signal : float
            Maximum signal to plot
        x_label : str
            Label for x-axis
        y_label : str
            Label for y-axis
        title : str
            Title for plot
    """

    ci_setup_pred = ci_setup_pred[ci_setup_true.isna().sum(axis=1) == 0]
    ci_setup_true = ci_setup_true[ci_setup_true.isna().sum(axis=1) == 0]

    ax.plot(ci_setup_true['mean'], color='b')
    ax.fill_between(ci_setup_true.index, ci_setup_true['lb'], ci_setup_true['ub'], color='b', alpha=.2)

    ax.plot(ci_setup_pred['mean'], color='r')
    ax.fill_between(ci_setup_pred.index, ci_setup_pred['lb'], ci_setup_pred['ub'], color='r', alpha=.2)
    ax.set_xlim((min_time, max_time))
    ax.set_ylim((min_signal, max_signal))
    trial_num = ci_setup_true['size'].max()
    ax.title.set_text(f'{title} — {trial_num} Trials')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()

    return

def get_time_alignment(x_label, adjusted_time, min_time=None, max_time=None, binsize=None):
    """
    Get the time alignment for the data
    Args:
        x_label : str
            Label for x-axis
        adjusted_time : pd.Series
            Time series with adjusted time
        min_time : float
            Minimum time to plot
        max_time : float
            Maximum time to plot
        binsize : float
            Size of bins to use
    Returns:
        plot_time : np.ndarray
            Adjusted time at each point
        x_label : str
            Label for x-axis
        min_time : float
            Minimum time to plot (adjusted to ms if provided)
        max_time : float
            Maximum time to plot (adjusted to ms if provided)
        binsize : float
            Size of bins to use (adjusted to ms if provided)
    """

    if binsize is not None:
        min_time *= binsize
        max_time *= binsize
        x_label = x_label.replace(' __', ' (ms)')
        plot_time = adjusted_time * binsize
    else:
        x_label = x_label.replace(' __', '')
        plot_time = adjusted_time
    
    return plot_time, x_label, min_time, max_time

    # tmp_backup['plot_time'] = tmp_backup['adjusted_time'] * binsize

def get_triplicated_data_for_time_alignment(df, alignment_col):
    """

    """

    rel_points = df[df[alignment_col] > 0].reset_index()
    identifiers = rel_points[['index', 'nTrial', 'nEndTrial']].dropna().values.astype(int)
    if len(identifiers) == 0:
        return pd.DataFrame(columns=df.reset_index().columns)

    # print('identifiers')
    # print(identifiers)
    
    lst_extendeds = []
    
    for idx, nTrial, nEndTrial in identifiers:
        extended_trial = df[(df['nTrial'] == nTrial) | (df['nEndTrial'] == nEndTrial)].reset_index().copy()
        extended_trial = extended_trial[(extended_trial['nTrial'] - extended_trial['nEndTrial']) == extended_trial['diffTrialNums']]
        extended_trial['index'] -= idx

        # print('extended_trial')
        # print(extended_trial)

        lst_extendeds.append(extended_trial.copy())
        
    relative_df = pd.concat(lst_extendeds)
    return relative_df

def plot_single_avg_reconstruction_v2(df, alignment_col, channel,
                                      title=None, trial_num=None, x_label=None, y_label=None,
                                      inx_bounds=(-40, 60), signal_bounds=(-1, 2.5),
                                      ic=None, color_lst=['b', 'g', 'y', 'k'],
                                      fig=None, ax=None, show_pred=True):
    """
    
    """

    color = color_lst[ic]
    
    relative_df = get_triplicated_data_for_time_alignment(df, alignment_col)

    df_filt_to_bounds = relative_df[relative_df['index'].between(*inx_bounds)].copy()
    df_filt_to_bounds['resids'] = df_filt_to_bounds[channel] - df_filt_to_bounds['pred']
    num_trials = (df_filt_to_bounds['index'] == 0).sum()

    rmse = np.sqrt((df_filt_to_bounds['resids']**2).mean())

    alignment_name = alignment_col.split('_')[-1]
    sns.lineplot(x='index', y=channel, data=df_filt_to_bounds, label=f'{alignment_col} — {channel} — True', ax=ax, color=color)
    # print('channel', channel, 'alignment_col', alignment_col, 'show_pred', show_pred)
    # print('channel', channel)
    if show_pred:
        sns.lineplot(x='index', y='pred', data=df_filt_to_bounds, label=f'{alignment_col} — {channel} — Pred', ax=ax, color='r')
    
    

    # fig.suptitle(f'{alignment_name} — {channel}')
    
    # ax.title.set_text(f'{title} — {trial_num} Trials')
    # ax.title.set_text(f'{title} — {trial_num} Trials — RMSE: {rmse:.2f}')
    ax.title.set_text(f'{title} — {num_trials} Trials — RMSE: {rmse:.2f}')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_ylim(*signal_bounds)
    ax.grid(visible=True)
    
    return

def plot_avg_reconstructions_v2(df,
                                # alignment_col_lst=['photometrySideInIndexr', 'photometryCenterInIndexr',
                                #                    'photometrySideInIndexnr','photometryCenterInIndexnr'],

                                alignment_col_lst=[
                                                   'photometrySideInIndexAA', 'photometrySideInIndexAa',
                                                   'photometrySideInIndexaA','photometrySideInIndexaa',
                                                   'photometrySideInIndexAB', 'photometrySideInIndexAb',
                                                   'photometrySideInIndexaB','photometrySideInIndexab',

                                                   'photometrySideOutIndexAA', 'photometrySideOutIndexAa',
                                                   'photometrySideOutIndexaA', 'photometrySideOutIndexaa',
                                                   'photometrySideOutIndexAB', 'photometrySideOutIndexAb',
                                                   'photometrySideOutIndexaB', 'photometrySideOutIndexab',
                                                   ],
                                
                                channel='zsgdFF',
                                channels=None,
                                plot_width=4,
                                binsize = 54,
                                min_time = -40, max_time = 60,
                                min_signal = -3.0, max_signal = 3.0,
                                title='Average Photometry Response Aligned to Side Port Entry — Holdout Data Only',
                                file_name=None,
                                save_data=None
                                ):
    """
    
    """
    inx_bounds = (min_time, max_time)
    signal_bounds = (min_signal, max_signal)

    x_label = 'Timesteps __ from Event'
    y_label = 'Response'

    fig, ax = plt.subplots(len(alignment_col_lst)//plot_width + (len(alignment_col_lst)%plot_width > 0)*1, plot_width)

    
    # # fig, ax = plt.subplots(2,2)
    # # max_i = len(alignment_col_lst)//plot_width
    # # max_j = 1
    
    
    # # plt.figure(figsize=(10,5))
    # fig, ax = plt.subplots(max_i + len(alignment_col_lst)%2, max_j+1)

    fig.suptitle(title)
    fig.set_figheight(20)
    fig.set_figwidth(40)

    if channels is None:
        channels = [channel]
    for ic, channel in enumerate(channels):
        for ialignment_col, alignment_col in enumerate(alignment_col_lst):

            show_pred = channels[channel] if len(channels) > 1 else True
            
            i,j = ialignment_col//plot_width, ialignment_col%plot_width
            plot_single_avg_reconstruction_v2(df, alignment_col, channel,
                                            #   title=f'{alignment_col} — ',
                                            title=f'{channel} - {alignment_col} Trials',
                                            inx_bounds=inx_bounds, signal_bounds=signal_bounds, fig=fig, ax=ax[i,j],
                                            ic=ic,
                                            show_pred=show_pred)


    
    if save_data is not None:
        for ialignment_col, alignment_col in enumerate(alignment_col_lst):
            i,j = ialignment_col//plot_width, ialignment_col%plot_width
            for line in ax[i, j].lines:
                lbl = line.get_label()
                print('i',i,'j',j,'—',lbl)
                save_fn = str(Path(save_data) / f'XY--{i}_{j}--{lbl}.npy')
                xy_dat = line.get_xydata()
                print(xy_dat)
                print(f'{save_fn}')
                with open(save_fn, 'wb') as save_obj:
                    np.save(save_obj, xy_dat)


        # ax[i, j].legend(['Mean Photometry Response',
        #                 'Predicted Photometry Response',
        #                 '95% SEM Confidence Interval'])
    ax[i, j].legend()
    fig.show()
    
    plt.tight_layout()
    fig.patch.set_facecolor('white')

    if file_name:
        fig.savefig(file_name)

    return
