#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  pyscript.py
#  
#  Copyright 2020 Kumaresh <kumaresh_krishnan@g.harvard.edu>
#
#  version 1.0
#  
import os, sys
import numpy as np

import hdf5storage as hdf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic, sem, spearmanr

import path
import pickle

def findBouts(raw_data, stimulus, pre, post):

    start = f'bouts_start_stimulus_{stimulus:03d}'
    end = f'bouts_end_stimulus_{stimulus:03d}'

    errors = raw_data[f'raw_stimulus_{stimulus:03d}']['errorcode']
    raw_time = raw_data[f'raw_stimulus_{stimulus:03d}']['timestamp']

    pos_x = raw_data[start]['fish_position_x']
    pos_y = raw_data[start]['fish_position_y']
    rpos_x = raw_data[f'raw_stimulus_{stimulus:03d}']['fish_position_x']
    rpos_y = raw_data[f'raw_stimulus_{stimulus:03d}']['fish_position_y']

    lim1, lim2 = 150-pre, 150 # For baseline bout rate
    
    window = (raw_time > lim1) & (raw_time < lim2) 
    window_cut = window & (rpos_x**2 + rpos_y**2 < 0.81)
    cut = window_cut.sum() / window.sum()
    window_errors = errors[window_cut]
    valid = (window_errors == 0).sum() / window_errors.shape[0]
    
    bout_stamps = raw_data[start]['timestamp']
    count = (bout_stamps > lim1) & (bout_stamps < lim2) & (pos_x**2 + pos_y**2 < 0.81)

    bout_rate = np.around(count.sum() / (valid*cut*(lim2-lim1)/60), 2) # 2 minutes window

    lim1, lim2 = 150, 150+post # Now for OMR performance
    #lim1, lim2 = 150-pre, 150 # Baseline performance
    #lim1 = np.random.randint(150-pre, 120) # This is to find 30 second random window to match stimulus duration
    #lim2 = lim1 + 30
    
    window = (raw_time > lim1) & (raw_time < lim2) 
    window_cut = window & (rpos_x**2 + rpos_y**2 < 0.81)
    cut = window_cut.sum() / window.sum()
    window_errors = errors[window_cut]

    valid = (window_errors == 0).sum() / window_errors.shape[0]
    omr_bouts = (bout_stamps > lim1) & (bout_stamps < lim2) & (pos_x**2 + pos_y**2 < 0.81)
    orig_bouts = (bout_stamps > lim1) & (bout_stamps < lim2)
    bout_cut = omr_bouts.sum() / orig_bouts.sum()

    stim_rate = np.around(omr_bouts.sum() / (valid*cut*(lim2-lim1)/60), 2)
 
    if valid < 0.9 or np.isnan(stim_rate):# or cut < 0.95:
        performance = np.nan
        incorrect_data = np.nan
        stim_rate = np.nan
        bout_rate = np.nan
    elif omr_bouts.sum() <= 0:
        performance = np.nan
        incorrect_data = omr_bouts.sum()
    else:
        angles = raw_data[start]['fish_accumulated_orientation'] \
            - raw_data[end]['fish_accumulated_orientation']

        valid_angles = angles[omr_bouts]

        if stimulus == 0:
            performance = (valid_angles < 0).sum() + 0.5*(valid_angles==0).sum()
        else:
            performance = (valid_angles > 0).sum() + 0.5*(valid_angles==0).sum()

        performance = np.around(performance / valid_angles.shape[0], 2)
        incorrect_data = valid_angles.shape[0]
            
    return bout_rate, performance, incorrect_data, stim_rate, cut*valid, bout_cut

def extractData(experiment, root, pre, post):

    info_path = path.Path() / '..' / experiment

    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()
    days = info['days']
    fish = info['fish']
    trials = info['trials']
    folders = info['folders']
    stimuli = info['stimuli']
    total_fish = np.sum(fish)

    fish_ctr = 0

    data_bout_rate = np.zeros((total_fish, stimuli, trials))
    data_performance = np.zeros((total_fish, stimuli, trials))
    data_incorrect_data = np.zeros((total_fish, stimuli, trials))
    data_stim_rate = np.zeros((total_fish, stimuli, trials))

    data_raw_cut = np.zeros((total_fish, stimuli, trials))
    data_bout_cut = np.zeros((total_fish, stimuli, trials))
    
    for day_idx, day in enumerate(days):

        for f in range(fish[day_idx]):

            for t in range(trials):
                
                folder = root / folders[day_idx] / f'{day}_fish{f+1:03d}' \
                    / 'raw_data' / f'trial{t:03d}.dat'
                tmp = open(folder, 'rb')
                raw_data = pickle.load(tmp)

                for stimulus in range(stimuli):    
                    bout_rate, performance, valids, stim_rate, cut, b_cut = findBouts(raw_data, stimulus, pre, post)
                    data_bout_rate[fish_ctr, stimulus, t] = bout_rate
                    data_performance[fish_ctr, stimulus, t] = performance
                    data_incorrect_data[fish_ctr, stimulus, t] = valids
                    data_stim_rate[fish_ctr, stimulus, t] = stim_rate
                    data_raw_cut[fish_ctr, stimulus, t] = cut
                    data_bout_cut[fish_ctr, stimulus, t] = b_cut

            fish_ctr += 1
                    
        print(day, fish_ctr, 'fish done')
        
    return data_bout_rate, 2*data_performance - 1, data_incorrect_data, data_stim_rate, data_raw_cut, data_bout_cut

def processData(experiment, data_bout_rate, data_performance, data_incorrect_data, data_stim_rate, data_raw_cut, data_bout_cut, mutant=False):

    info_path = path.Path() / '..' / experiment

    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()

    if mutant:
        filt = info['filt']
    else:
        filt = np.arange(data_bout_rate.shape[0])

    to_save = {}

    to_save['bout_rate'] = data_bout_rate[filt]
    to_save['performance'] = data_performance[filt]
    to_save['incorrect_data'] = data_incorrect_data[filt]
    to_save['stim_rate'] = data_stim_rate[filt]
    to_save['raw_cut'] = data_raw_cut[filt]
    to_save['bout_cut'] = data_bout_cut[filt]
    
    return to_save

def main(experiment, pre, post, mutant):

    root = path.Path() / '..'
    
    data_bout_rate, data_performance, data_incorrect_data, data_stim_rate, data_raw_cut, data_bout_cut \
        = extractData(experiment, root, pre, post)

    to_save = processData(experiment, data_bout_rate, data_performance, \
        data_incorrect_data, data_stim_rate, data_raw_cut, data_bout_cut, mutant)

    save_dir = path.Path() / '..' / experiment / f'data_bout_rate_pre_{pre}_post_{post}'
    hdf.savemat(save_dir, to_save, format='7.3', oned_as='column', store_python_metadata=True)

    return 0

def plotData(experiment, pre, post, model):

    data_path = path.Path() / '..' / experiment
    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    
    save_dir = path.Path() / '..' / experiment / \
        f'individual_performance_pre_{pre}_post_{post}'
    os.makedirs(save_dir, exist_ok=True)

    bout_rate = tmp['bout_rate']
    performance = tmp['performance']

    corrs = []

    # for row in range(bout_rate.shape[0]):
        
    #     f, ax = plt.subplots()

    #     valid = ~np.isnan(bout_rate[row,:,:]) & ~np.isnan(performance[row,:,:])

    #     stats = spearmanr(bout_rate[row,:,:][valid].ravel(), performance[row,:,:][valid].ravel()) # 15 bins, each size 10

        
    #     ax.scatter(bout_rate[row,:,:].ravel(), performance[row,:,:].ravel(), \
    #         label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}', color=coloring)

    #     corrs.append(stats[0])

    #     ax.set_xlabel('Bout rate (bouts/min)')
    #     ax.set_ylabel('Performance')
    #     #ax.set_title('Bout rate vs performance')
    #     ax.grid(False)
    #     ax.legend()
    #     ax.set_xlim(0,150)
    #     ax.set_ylim(-1.1,1.1)
    #     sns.despine(top=True, right=True)

    #     f.savefig(save_dir / f'scatter_fish_{row}.pdf')
    #     f.savefig(save_dir / f'scatter_fish_{row}.png')
    #     plt.close(f)

    # f, ax = plt.subplots()

    # vals, bins = np.histogram(corrs, range=(-1,1), bins=20)
    # bins = 0.5*(bins[1:] + bins[:-1])

    # ax.plot(bins, vals, marker='o', color=coloring)
    # ax.set_xlabel('Spearman r value')
    # ax.set_ylabel('Count')
    # #ax.set_title('Distribution of r values across fish')
    # ax.grid(False)
    # sns.despine(top=True, right=True)

    # f.savefig(save_dir / 'corr_distribution.pdf')
    # f.savefig(save_dir / 'corr_distribution.png')

    # plt.close(f)

    # Full plot of all fish, all trials, all raw_data scattered

    #states = np.where(tmp['state'] == 0)
    #states = tmp['state']
    #ax.scatter(bout_rate[states].ravel(), performance[states].ravel(), alpha=0.05)

    valid = ~np.isnan(bout_rate) & ~np.isnan(performance) #& (states == 1)

    stats = spearmanr(bout_rate[valid].ravel(), performance[valid].ravel()) # 15 bins, each size 10

    f, ax = plt.subplots()
        
    ax.scatter(bout_rate[:,:,:].ravel(), performance[:,:,:].ravel(), alpha=0.05, label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}', color=coloring)
    #ax.scatter(bout_rate[valid].ravel(), performance[valid].ravel(), alpha=0.05, label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}')

    ax.set_xlabel('Baseline bout rate (bouts/min)')
    ax.set_ylabel('Performance')
    ##ax.set_title('Baseline bout rate vs performance')
    ax.legend()
    ax.grid(False)
    ax.set_xlim(0,150)
    ax.set_ylim(-1.1,1.1)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'raw_scatter.pdf')
    f.savefig(save_dir / f'raw_scatter.png')
    plt.close(f)

    valid = ~np.isnan(bout_rate)
    bout_rate = bout_rate[valid]
    performance = performance[valid]

    means, _, _ = binned_statistic(bout_rate, performance, statistic=np.nanmean, range=(0,150), bins=15)
    sems, _, _ = binned_statistic(bout_rate, performance, statistic=lambda x: sem(x, nan_policy='omit'), range=(0,150), bins=15)

    xrates = np.arange(5,150,10)
    valid = ~np.isnan(means[1:])
    
    stats = spearmanr(xrates[1:][valid], means[1:][valid]) # 15 bins, each size 10

    bout_rate = tmp['bout_rate'] # Stupidly modified above, so reloading
    performance = tmp['performance'] # Stupidly modified above, so reloading
        
    f, ax = plt.subplots()
    
    ax.errorbar(xrates, means, yerr=sems, marker='o', ecolor='black', color=coloring,\
        linestyle='None', capsize=4.0, label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}')

    ax.set_xlabel('Baseline bout rate (bouts/min)')
    ax.set_ylabel('Avg performance')
    ##ax.set_title('Baseline bout rate vs performance')
    ax.legend()
    ax.grid(False)
    ax.set_xlim(0,150)
    ax.set_ylim(-1.1,1.1)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'scatter_binned_corr.pdf')
    f.savefig(save_dir / f'scatter_binned_corr.png')
    plt.close(f)

    # Fishwise plotting

    bout_rate = tmp['bout_rate'] # Stupidly modified above, so reloading
    performance = tmp['performance'] # Stupidly modified above, so reloading
    
    dims = bout_rate.shape
    fish_rate = np.nanmean(bout_rate, axis=(1,2))
    fish_perf = np.nanmean(performance, axis=(1,2))
    fish_rate_sem = sem(bout_rate.reshape(dims[0],-1), axis=1, nan_policy='omit')
    fish_perf_sem = sem(performance.reshape(dims[0], -1), axis=1, nan_policy='omit')

    valid = ~np.isnan(fish_rate)
    stats = spearmanr(fish_rate[valid], fish_perf[valid]) # 15 bins, each size 10

    f, ax = plt.subplots()
    
    _, c, b = ax.errorbar(fish_rate, fish_perf, xerr=fish_rate_sem, yerr=fish_perf_sem, marker='o', ecolor='black',\
        markersize=4.0, linestyle='None', capsize=4.0, label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}', alpha=0.6, color=coloring)

    [val.set_alpha(0.2) for val in c]
    [val.set_alpha(0.2) for val in b]

    ax.set_xlabel('Bout rate (bouts/min)')
    ax.set_ylabel('Avg performance')
    #ax.set_title('Bout rate vs performance fishwise')
    ax.legend()
    ax.grid(False)
    ax.set_xlim(0,150)
    ax.set_ylim(-1.1,1.1)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'fish_binned_corr.pdf')
    f.savefig(save_dir / f'fish_binned_corr.png')
    plt.close(f)

    return 0

def plotStimRateCorrelation(experiment, pre, post, model):

    data_path = path.Path() / '..' / experiment
    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    
    save_dir = path.Path() / '..' / experiment / \
        f'individual_stim_rate_correlation_pre_{pre}_post_{post}'
    os.makedirs(save_dir, exist_ok=True)

    performance = tmp['performance']
    rates = tmp['stim_rate']

    # for row in range(performance.shape[0]):
        
    #     f, ax = plt.subplots()
        
    #     ax.scatter(rates[row,:,:].ravel(), performance[row,:,:].ravel(), color=coloring)

    #     ax.set_xlabel('Stimulus bout rate')
    #     ax.set_ylabel('Performance')
    #     #ax.set_title('Stimulus bout rate')
    #     ax.grid(False)
    #     ax.set_xlim(0,150)
    #     ax.set_ylim(-1.1,1.1)
    #     sns.despine(top=True, right=True)

    #     f.savefig(save_dir / f'stim_rate_correlation_{row}.pdf')
    #     f.savefig(save_dir / f'stim_rate_correlation_{row}.png')
    #     plt.close(f)

    valid = ~np.isnan(rates) & ~np.isnan(performance)

    stats = spearmanr(rates[valid].ravel(), performance[valid].ravel()) # 15 bins, each size 10

    # Full plot of all fish, all trials, all raw_data scattered
    f, ax = plt.subplots()
        
    ax.scatter(rates[:,:,:].ravel(), performance[:,:,:].ravel(), alpha=0.05, label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}', color=coloring)

    ax.set_xlabel('Stimulus bout rate (bouts/min)')
    ax.set_ylabel('Performance')
    ##ax.set_title('Stimulus bout rate vs. performance')
    ax.legend()
    ax.grid(False)
    ax.set_xlim(0,150)
    ax.set_ylim(-1.1,1.1)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'raw_stim_rate_correlation.pdf')
    f.savefig(save_dir / f'raw_stim_rate_correlation.png')
    plt.close(f)

    # Binned version of above

    performance = tmp['performance'] # Stupidly modified above, so reloading
    rates = tmp['stim_rate'] # Stupidly modified above, so reloading

    valid = ~np.isnan(rates)
    rates = rates[valid]
    performance = performance[valid]

    means, _, _ = binned_statistic(rates, performance, statistic=np.nanmean, range=(0,150), bins=15)
    sems, _, _ = binned_statistic(rates, performance, statistic=lambda x: sem(x, nan_policy='omit'), range=(0,150), bins=15)

    xrates = np.arange(5,150,10)
    valid = ~np.isnan(means[1:])
    
    stats = spearmanr(xrates[1:][valid], means[1:][valid]) # 15 bins, each size 10

    f, ax = plt.subplots()
        
    _, c, b = ax.errorbar(xrates, means, yerr=sems, marker='o', ecolor='black',\
        linestyle='None', capsize=4.0, label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}', color=coloring)

    ax.set_xlabel('Stimulus bout rate (bouts/min)')
    ax.set_ylabel('Avg Performance')
    ##ax.set_title('Stimulus bout rate vs. performance')
    ax.legend()
    ax.grid(False)
    ax.set_xlim(0,150)
    ax.set_ylim(-1.1,1.1)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'binned_stim_rate_correlation.pdf')
    f.savefig(save_dir / f'binned_stim_rate_correlation.png')
    plt.close(f)

    # Fishwise plotting

    performance = tmp['performance'] # Stupidly modified above, so reloading
    rates = tmp['stim_rate'] # Stupidly modified above, so reloading

    dims = rates.shape
    fish_rate = np.nanmean(rates, axis=(1,2))
    fish_perf = np.nanmean(performance, axis=(1,2))
    fish_rate_sem = sem(rates.reshape(dims[0],-1), axis=1, nan_policy='omit')
    fish_perf_sem = sem(performance.reshape(dims[0], -1), axis=1, nan_policy='omit')

    valid = ~np.isnan(fish_rate)
    stats = spearmanr(fish_rate[valid], fish_perf[valid]) # 15 bins, each size 10

    f, ax = plt.subplots()
    
    _, c, b = ax.errorbar(fish_rate, fish_perf, xerr=fish_rate_sem, yerr=fish_perf_sem, marker='o', ecolor='black',\
        markersize=4.0, linestyle='None', capsize=4.0, label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}', alpha=0.6, color=coloring)

    [val.set_alpha(0.2) for val in c]
    [val.set_alpha(0.2) for val in b]

    ax.set_xlabel('Stimulus bout rate (bouts/min)')
    ax.set_ylabel('Avg performance')
    #ax.set_title('Stimulus bout rate vs performance fishwise')
    ax.legend()
    ax.grid(False)
    ax.set_xlim(0,150)
    ax.set_ylim(-1.1,1.1)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'fish_binned_corr.pdf')
    f.savefig(save_dir / f'fish_binned_corr.png')
    plt.close(f)

    return 0

def plotStimBaselineCorrelation(experiment, pre, post, model):

    data_path = path.Path() / '..' / experiment
    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    
    save_dir = path.Path() / '..' / experiment / \
        f'individual_stim_baseline_correlation_pre_{pre}_post_{post}'
    os.makedirs(save_dir, exist_ok=True)

    b_rates = tmp['bout_rate']
    rates = tmp['stim_rate']

    for row in range(b_rates.shape[0]):
        
        f, ax = plt.subplots()
        
        ax.scatter(b_rates[row,:,:].ravel(), rates[row,:,:].ravel(), color=coloring)

        ax.set_xlabel('Baseline bout rate')
        ax.set_ylabel('Stimulus bout rate')
        ##ax.set_title('Stimulus bout rate vs Baseline bout rate')
        ax.grid(False)
        ax.set_xlim(0,150)
        ax.set_ylim(0,150)
        sns.despine(top=True, right=True)

        f.savefig(save_dir / f'stim_base_correlation_{row}.pdf')
        f.savefig(save_dir / f'stim_base_correlation_{row}.png')
        plt.close(f)

    valid = ~np.isnan(rates) & ~np.isnan(b_rates)

    stats = spearmanr(b_rates[valid].ravel(), rates[valid].ravel()) # 15 bins, each size 10

    # Full plot of all fish, all trials, all raw_data scattered
    f, ax = plt.subplots()
        
    ax.scatter(b_rates[:,:,:].ravel(), rates[:,:,:].ravel(), alpha=0.05, label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}', color=coloring)

    ax.set_xlabel('Baseline bout rate')
    ax.set_ylabel('Stimulus bout rate')
    ##ax.set_title('Stimulus bout rate vs. Baseline bout rate')
    ax.legend()
    ax.grid(False)
    ax.set_xlim(0,150)
    ax.set_ylim(0,150)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'raw_stim_base_correlation.pdf')
    f.savefig(save_dir / f'raw_stim_base_correlation.png')
    plt.close(f)

    # Binned version of above

    b_rates = tmp['bout_rate'] # Stupidly modified above, so reloading
    rates = tmp['stim_rate'] # Stupidly modified above, so reloading

    valid = ~np.isnan(b_rates)
    b_rates = b_rates[valid]
    rates = rates[valid]

    means, _, _ = binned_statistic(b_rates, rates, statistic=np.nanmean, range=(0,150), bins=15)
    sems, _, _ = binned_statistic(b_rates, rates, statistic=lambda x: sem(x, nan_policy='omit'), range=(0,150), bins=15)

    xrates = np.arange(5,150,10)
    valid = ~np.isnan(means[1:])
    
    stats = spearmanr(xrates[1:][valid], means[1:][valid]) # 15 bins, each size 10

    f, ax = plt.subplots()
        
    ax.errorbar(xrates, means, yerr=sems, marker='o', ecolor='black',\
        linestyle='None', capsize=4.0, label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}', color=coloring)

    ax.set_xlabel('Baseline bout rate')
    ax.set_ylabel('Stimulus bout rate')
    ##ax.set_title('Stimulus bout rate vs. Baseline bout rate')
    ax.legend()
    ax.grid(False)
    ax.set_xlim(0,150)
    ax.set_ylim(0,150)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'binned_stim_base_correlation.pdf')
    f.savefig(save_dir / f'binned_stim_base_correlation.png')
    plt.close(f)

    # Fishwise plotting

    b_rates = tmp['bout_rate'] # Stupidly modified above, so reloading
    rates = tmp['stim_rate'] # Stupidly modified above, so reloading

    dims = rates.shape
    fish_rate = np.nanmean(rates, axis=(1,2))
    fish_b_rate = np.nanmean(b_rates, axis=(1,2))
    fish_rate_sem = sem(rates.reshape(dims[0],-1), axis=1, nan_policy='omit')
    fish_b_rate_sem = sem(b_rates.reshape(dims[0], -1), axis=1, nan_policy='omit')

    valid = ~np.isnan(fish_b_rate)
    stats = spearmanr(fish_b_rate[valid], fish_rate[valid]) # 15 bins, each size 10

    f, ax = plt.subplots()
    
    _, c, b = ax.errorbar(fish_b_rate, fish_rate, xerr=fish_b_rate_sem, yerr=fish_rate_sem, marker='o', ecolor='black',\
        markersize=4.0, linestyle='None', capsize=4.0, label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}', alpha=0.6, color=coloring)

    [val.set_alpha(0.2) for val in c]
    [val.set_alpha(0.2) for val in b]

    ax.set_xlabel('Baseline bout rate (bouts/min)')
    ax.set_ylabel('Stimulus bout rate (bouts/min)')
    #ax.set_title('Stimulus bout rate vs baseline bout rate fishwise')
    ax.legend()
    ax.grid(False)
    ax.set_xlim(0,150)
    ax.set_ylim(0,150)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'fish_binned_corr.pdf')
    f.savefig(save_dir / f'fish_binned_corr.png')
    plt.close(f)

    return 0

def quickie(experiment, pre, post, model):

    data_path = path.Path() / '..' / experiment
    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    
    save_dir = path.Path() / '..' / experiment

    b_rates = tmp['bout_rate'] # stim rate or bout rate

    br = tmp['stim_rate']
    #print(np.nanmean(b_rates), sem(b_rates.ravel(), nan_policy='omit'), np.nanmean(br), sem(br.ravel(), nan_policy='omit')); input()
    f, ax = plt.subplots()

    vals, bins = np.histogram(b_rates.ravel(), range=(0,150), bins=15)
    vals_stim, bins = np.histogram(br.ravel(), range=(0,150), bins=15)
    bins = 0.5*(bins[1:] + bins[:-1])

    vals = vals / vals.sum()
    vals_stim = vals_stim / vals_stim.sum()

    ax.plot(bins, vals, marker='^', label='baseline', color=coloring, alpha=0.3, linestyle='--', markersize=4)
    ax.plot(bins, vals_stim, marker='o', label='stimulus', color=coloring, markersize=4)
    ax.set_xlabel('Bout rate (bouts/min)')
    ax.set_ylabel('Normalized count')
    ##ax.set_title('Distribution of bout rates')
    ax.set_ylim(0,1.1)
    #ax.legend()
    ax.grid(False)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / 'bout_distribution.pdf', transparent=True)
    f.savefig(save_dir / 'bout_distribution.png', transparent=True)

    plt.close()

    return

model = False
coloring = 'red' if model else 'black'

if __name__ == '__main__':

    # !!!!!!!!!!!!!!!!! If looking at baseline then change lim1/lim2 accordingly !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    s = 15
    sns.set_style('white')
    sns.set_style('ticks')
    plt.rc('font', size=s)          # controls default text sizes
    plt.rc('axes', titlesize=s)     # fontsize of the axes title
    plt.rc('axes', labelsize=s)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=s)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=s)    # fontsize of the tick labels
    plt.rc('legend', fontsize=s-2)    # legend fontsize
    plt.rc('figure', titlesize=s)  # fontsize of the figure title
    #plt.rc('figure', figsize=(6,4))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.facecolor'] = (0,0,0,0)
    plt.rcParams['axes.facecolor'] = (0,0,0,0)
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['axes.linewidth'] = 0.25
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = False
    
    pairs = [(120,30)]

    mutant = False # If mutant, have to load genotype information differently, only for main()
    experiments = ['experimental_analysis']
    #experiments = ['simulation']

    for experiment in experiments:
        for pre, post in pairs:

            main(experiment, pre, post, mutant) # comment for modeling
            plotData(experiment, pre, post, model)
            plotStimRateCorrelation(experiment, pre, post, model)
            plotStimBaselineCorrelation(experiment, pre, post, model)
            quickie(experiment, pre, post, model)
    
    sys.exit(0)
