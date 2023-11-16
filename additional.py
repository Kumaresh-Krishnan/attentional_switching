
import numpy as np
import hdf5storage as hdf
import matplotlib.pyplot as plt
import seaborn as sns
import path
import os
from scipy.stats import linregress, spearmanr, sem, binned_statistic, binom
from sklearn import mixture
from sklearn.model_selection import GridSearchCV
import pandas as pd
from scipy.optimize import curve_fit
from scipy import interpolate as ip

from matplotlib import cm
from matplotlib.colors import ListedColormap
maps = cm.get_cmap('viridis_r', 256)
new_map = ListedColormap(maps(np.linspace(0,0.7,256)))

def removeNan(arr):

    return ~np.isnan(arr)

def groupTFB(experiment, pre, post):

    data_path = path.Path() / '..' / experiment
    tmp = hdf.loadmat(data_path / f'data_tfb')

    tfb = tmp['tfb']
    #tfb_performance = tmp['tfb_performance']

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    b_rates = tmp['bout_rate']
    br = tmp['bout_rate']
    performance = tmp['performance']

    avg = br.mean(axis=(1,2))
    ranks = np.argsort(avg)
    #print(avg)

    g1 = ranks[:16]
    g2 = ranks[16:-16]
    g3 = ranks[-16:]

    b_rates_1 = b_rates[g1]
    b_rates_2 = b_rates[g2]
    b_rates_3 = b_rates[g3]

    tfb_1 = tfb[g1].ravel()
    tfb_2 = tfb[g2].ravel()
    tfb_3 = tfb[g3].ravel()

    perf_1 = performance[g1].ravel()
    perf_2 = performance[g2].ravel()
    perf_3 = performance[g3].ravel()

    print(np.nanmean(perf_1))
    print(np.nanmean(perf_2))
    print(np.nanmean(perf_3))

    valid = ~np.isnan(tfb_1); tfb_1 = tfb_1[valid]; perf_1 = perf_1[valid]
    valid = ~np.isnan(tfb_2); tfb_2 = tfb_2[valid]; perf_2 = perf_2[valid]
    valid = ~np.isnan(tfb_3); tfb_3 = tfb_3[valid]; perf_3 = perf_3[valid]

    vals_1, bins, _ = binned_statistic(tfb_1, perf_1, statistic=np.nanmean, range=(0,2), bins=20)
    errors_1, _, _ = binned_statistic(tfb_1, perf_1, statistic=lambda x: sem(x, nan_policy='omit'), range=(0,2), bins=20)

    vals_2, bins, _ = binned_statistic(tfb_2, perf_2, statistic=np.nanmean, range=(0,2), bins=20)
    errors_2, _, _ = binned_statistic(tfb_2, perf_2, statistic=lambda x: sem(x, nan_policy='omit'), range=(0,2), bins=20)

    vals_3, bins, _ = binned_statistic(tfb_3, perf_3, statistic=np.nanmean, range=(0,2), bins=20)
    errors_3, _, _ = binned_statistic(tfb_3, perf_3, statistic=lambda x: sem(x, nan_policy='omit'), range=(0,2), bins=20)

    bins = 0.5*(bins[1:] + bins[:-1])

    
    
    '''
    f, ax = plt.subplots()
    '''
    #ax.errorbar(bins, vals_1, yerr=errors_1, ecolor='gray', capsize=2.0, label='bottom 16')
    #ax.errorbar(bins, vals_2, yerr=errors_2, ecolor='gray', capsize=2.0, label='middle 32')
    #ax.errorbar(bins, vals_3, yerr=errors_3, ecolor='gray', capsize=2.0, label='top 16')
    '''

    ax.plot(bins, vals_1, label='bottom 16')
    ax.plot(bins, vals_2, label='middle 16')
    ax.plot(bins, vals_3, label='top 16')

    ax.fill_between(bins, vals_1+errors_1, vals_1-errors_1, alpha=0.2)
    ax.fill_between(bins, vals_2+errors_2, vals_2-errors_2, alpha=0.2)
    ax.fill_between(bins, vals_3+errors_3, vals_3-errors_3, alpha=0.2)

    ax.legend()
    ax.grid(False)
    ax.set_ylim(0,1.1)
    sns.despine(top=True, right=True)

    f.savefig(data_path / 'group_tfb_distribution.png')
    f.savefig(data_path / 'group_tfb_distribution.pdf')

    plt.close(f)
    '''
    vals_1, bins = np.histogram(b_rates_1, range=(0,150), bins=10)
    vals_2, bins = np.histogram(b_rates_2, range=(0,150), bins=10)
    vals_3, bins = np.histogram(b_rates_3, range=(0,150), bins=10)
    bins = 0.5*(bins[1:] + bins[:-1])

    vals_1 = vals_1 / vals_1.sum()
    vals_2 = vals_2 / vals_2.sum()
    vals_3 = vals_3 / vals_3.sum()

    f, ax = plt.subplots()

    ax.plot(bins, vals_1, marker='o', label=f'bottom 16 (perf: {np.nanmean(perf_1):.2f})', color=coloring, linestyle='-')
    ax.plot(bins, vals_2, marker='o', label=f'middle 32 (perf: {np.nanmean(perf_2):.2f})', color=coloring, alpha=0.5)
    ax.plot(bins, vals_3, marker='o', label=f'top 16 (perf: {np.nanmean(perf_3):.2f})', color=coloring, linestyle='--')
    #ax.set_title('Distribution of bout rates across groups of fish based on avg bout rate')
    ax.legend()
    ax.set_xlabel('Bout rate (bout/min)')
    ax.set_ylabel('Normalized count')
    ax.set_ylim(0,1.1)
    ax.grid(False)
    sns.despine(top=True, right=True)

    f.savefig(data_path / 'group_distribution.png')
    f.savefig(data_path / 'group_distribution.pdf')

    plt.close(f)

    return

def explorePerf(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    performance = tmp['performance']
    #print(np.nanmean(performance)); input()
    lpc = performance[performance < 0.8]
    hpc = performance[performance >= 0.8]

    
    

    vl, bl = np.histogram(lpc, range=(-1,0.6), bins=16)
    bl = (bl[1:] + bl[:-1])*0.5

    vh, bh = np.histogram(hpc, range=(0.6,1), bins=4)
    bh = (bh[1:] + bh[:-1])*0.5

    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(bl, vl, marker='o')
    ax2.plot(bh, vh, marker='o')
    ax1.set_xlabel('Performance')
    ax1.set_ylabel('Count')
    ax2.set_xlabel('Performance')
    ax2.set_ylabel('Count')
    
    plt.suptitle('Distribution of performance scores')
    ax1.grid(False)
    ax2.grid(False)
    sns.despine(top=True, right=True, ax=ax1)
    sns.despine(top=True, right=True, ax=ax2)

    #plt.show()

    f.savefig(data_path / 'perf_distribution_filt.pdf')
    f.savefig(data_path / 'perf_distribution_filt.png')

    plt.close(f)

    v, b = np.histogram(performance, range=(-1,1), bins=20)
    b = (b[1:] + b[:-1])*0.5

    v = v / v.sum()

    f, ax = plt.subplots()

    ax.plot(b, v, marker='o', color=coloring)

    ax.set_xlabel('Performance')
    ax.set_ylabel('Count')
    #ax.set_title('Distribution of performance scores')
    ax1.grid(False)
    ax.set_ylim(0,0.6)
    
    sns.despine(top=True, right=True, ax=ax)

    f.savefig(data_path / 'perf_distribution.pdf')
    f.savefig(data_path / 'perf_distribution.png')

    plt.close(f)

    return

def individualPerf(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    performance = tmp['performance']

    thresh = 0.5

    engage = (performance > thresh).reshape(performance.shape[0], -1).sum(axis=1) / (performance.shape[1]*performance.shape[2])

    
    

    f, ax = plt.subplots(figsize=(12,8))

    ax.bar(range(1, performance.shape[0]+1), engage, label='engage')
    ax.bar(range(1, performance.shape[0]+1), 1-engage, bottom=engage, label='disengage')

    ax.set_xlabel('Fish number')
    ax.set_ylabel('Percentage of OMR engagement')
    #ax.set_title('Percentage of engagement in OMR across fish')
    ax.set_xticks(range(1, engage.shape[0]+1))
    ax.set_xticklabels([str(i) for i in range(1, engage.shape[0]+1)], rotation=90)
    ax.grid(False)
    ax.legend()
    sns.despine(top=True, right=True)

    f.savefig(data_path / 'engage_prop.pdf')
    f.savefig(data_path / 'engage_prop.png')

    plt.close(f)

    return

def perfOrder(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    performance = tmp['performance']

    performance = performance.reshape(performance.shape[0], -1)

    t_o = np.array([0,15,1,16,2,17,3,18,4,19,5,20,6,21,7,22,8,23, \
        9,24,10,25,11,26,12,27,13,28,14,29])
    new_idx = (np.tile(np.arange(performance.shape[0]), (performance.shape[1],1)).T, np.tile(t_o, (performance.shape[0],1)))

    performance = performance[new_idx]

    f, ax = plt.subplots()
    
    sns.heatmap(performance, vmin=-1.0, vmax=1.0, cmap='coolwarm' ,center=0, mask=np.isnan(performance))
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Fish number')
    #ax.set_title('Performance variation across trials for each fish')

    f.savefig(data_path / 'trial_performance.pdf')
    f.savefig(data_path / 'trial_performance.png')

    plt.close(f)

    trial_wise = np.nanmean(performance, axis=0)
    errs = sem(performance, axis=0, nan_policy='omit')

    f, ax = plt.subplots()

    ax.errorbar(range(1,trial_wise.shape[0]+1), trial_wise, yerr=errs, marker='o', capsize=2.0, ecolor='gray', color=coloring)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Avg. Performance')
    #ax.set_title('Performance variation with trial number')
    ax.set_ylim(-1.1,1.1)
    ax.set_xlim(0,40)
    ax.grid(False)
    sns.despine(top=True, right=True)

    f.savefig(data_path / 'avg_trial_performance.pdf')
    f.savefig(data_path / 'avg_trial_performance.png')

    plt.close(f)
    
    return

def rankedPerf(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    performance = tmp['performance']

    performance = performance.reshape(performance.shape[0], -1)

    avg = np.nanmean(performance, axis=1)
    idx = np.flip(np.argsort(avg))
    t_o = np.array([0,15,1,16,2,17,3,18,4,19,5,20,6,21,7,22,8,23,9,24,10,25,11,26,12,27,13,28,14,29])
    new_idx = (np.tile(idx, (performance.shape[1],1)).T, np.tile(t_o, (performance.shape[0],1)))

    performance = performance[new_idx]

    f, ax = plt.subplots()
    
    sns.heatmap(performance, vmin=-1.0, vmax=1.0, cmap=new_map, center=0, mask=np.isnan(performance))
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Fish number')
    ax.set_xticks(ax.get_xticks()[::2])
    #ax.set_xticklabels(ax.get_xticklabels()[::2])
    ax.set_yticks(ax.get_yticks()[::2])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    #ax.set_title('Ranked performance variation across trials for each fish')

    f.savefig(data_path / 'trial_performance_ranked.pdf')
    f.savefig(data_path / 'trial_performance_ranked.png')

    plt.close(f)

    return

def rankedRate(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    b_rates = tmp['bout_rate']
    performance = tmp['performance']

    b_rates = b_rates.reshape(b_rates.shape[0], -1)
    performance = performance.reshape(performance.shape[0], -1)

    avg = np.nanmean(performance, axis=1)
    idx = np.flip(np.argsort(avg))
    t_o = np.array([0,15,1,16,2,17,3,18,4,19,5,20,6,21,7,22,8,23,9,24,10,25,11,26,12,27,13,28,14,29])
    new_idx = (np.tile(idx, (b_rates.shape[1],1)).T, np.tile(t_o, (b_rates.shape[0],1)))

    b_rates = b_rates[new_idx]

    f, ax = plt.subplots()
    
    sns.heatmap(b_rates, vmin=0, vmax=120, cmap='coolwarm' ,center=60, mask=np.isnan(b_rates))
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Fish number')
    #ax.set_title('Ranked baseline bout rate variation across trials for each fish')

    f.savefig(data_path / 'trial_rate_ranked.pdf')
    f.savefig(data_path / 'trial_rate_ranked.png')

    plt.close(f)

    return

def rateOrder(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    b_rates = tmp['bout_rate']

    b_rates = b_rates.reshape(b_rates.shape[0], -1)

    t_o = np.array([0,15,1,16,2,17,3,18,4,19,5,20,6,21,7,22,8,23, \
        9,24,10,25,11,26,12,27,13,28,14,29])
    new_idx = (np.tile(np.arange(b_rates.shape[0]), (b_rates.shape[1],1)).T, np.tile(t_o, (b_rates.shape[0],1)))

    b_rates = b_rates[new_idx]

    f, ax = plt.subplots()
    
    sns.heatmap(b_rates, vmin=0, vmax=120, cmap='coolwarm' ,center=60, mask=np.isnan(b_rates))
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Fish number')
    #ax.set_title('Baseline bout rate variation across trials for each fish')

    f.savefig(data_path / 'trial_rate.pdf')
    f.savefig(data_path / 'trial_rate.png')

    plt.close(f)

    trial_wise = np.nanmean(b_rates, axis=0)
    errs = sem(b_rates, axis=0, nan_policy='omit')

    f, ax = plt.subplots()

    ax.errorbar(range(1,trial_wise.shape[0]+1), trial_wise, yerr=errs, marker='o', capsize=2.0, ecolor='gray', color=coloring)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Avg. Bout Rate')
    #ax.set_title('Baseline bout rate variation with trial number')
    ax.set_ylim(0,150)
    ax.grid(False)
    sns.despine(top=True, right=True)

    f.savefig(data_path / 'avg_trial_rate.pdf')
    f.savefig(data_path / 'avg_trial_rate.png')

    plt.close(f)
    
    return

def rankedStim(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    s_rates = tmp['stim_rate']
    performance = tmp['performance']

    s_rates = s_rates.reshape(s_rates.shape[0], -1)
    performance = performance.reshape(performance.shape[0], -1)

    avg = np.nanmean(performance, axis=1)
    idx = np.flip(np.argsort(avg))
    t_o = np.array([0,15,1,16,2,17,3,18,4,19,5,20,6,21,7,22,8,23,9,24,10,25,11,26,12,27,13,28,14,29])
    new_idx = (np.tile(idx, (s_rates.shape[1],1)).T, np.tile(t_o, (s_rates.shape[0],1)))

    s_rates = s_rates[new_idx]

    f, ax = plt.subplots()

    sns.heatmap(s_rates, vmin=0, vmax=160, cmap='coolwarm', mask=np.isnan(s_rates))
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Fish number')
    #ax.set_title('Ranked stimulus bout rate variation across trials for each fish')

    f.savefig(data_path / 'trial_stim_ranked.pdf')
    f.savefig(data_path / 'trial_stim_ranked.png')

    plt.close(f)

    return

def numBoutsOrder(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    numbouts = tmp['incorrect_data']

    numbouts = numbouts.reshape(numbouts.shape[0], -1)

    t_o = np.array([0,15,1,16,2,17,3,18,4,19,5,20,6,21,7,22,8,23, \
        9,24,10,25,11,26,12,27,13,28,14,29])
    new_idx = (np.tile(np.arange(numbouts.shape[0]), (numbouts.shape[1],1)).T, np.tile(t_o, (numbouts.shape[0],1)))

    numbouts = numbouts[new_idx]

    f, ax = plt.subplots()
    
    sns.heatmap(numbouts, vmin=0, vmax=np.nanmax(numbouts), cmap='coolwarm', mask=np.isnan(numbouts))
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Fish number')
    #ax.set_title('#Bouts variation across trials for each fish')

    f.savefig(data_path / 'trial_numbouts.pdf')
    f.savefig(data_path / 'trial_numbouts.png')

    plt.close(f)

    vals, bins = np.histogram(numbouts.ravel(), range=(0,60), bins=10)
    bins = 0.5*(bins[1:] + bins[:-1])

    f, ax = plt.subplots()

    ax.plot(bins, vals, marker='o')
    ax.set_xlabel('# Bouts')
    ax.set_ylabel('Count')
    #ax.set_title('Distribution of #Bouts used for performance')
    #ax.set_ylim(0,1.1)
    ax.grid(False)
    sns.despine(top=True, right=True)

    f.savefig(data_path / 'dist_numbouts.pdf')
    f.savefig(data_path / 'dist_numbouts.png')

    plt.close(f)
    
    return

def rankedNumBouts(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    numbouts = tmp['incorrect_data']
    performance = tmp['performance']

    numbouts = numbouts.reshape(numbouts.shape[0], -1)
    performance = performance.reshape(performance.shape[0], -1)

    avg = np.nanmean(performance, axis=1)
    idx = np.flip(np.argsort(avg))
    t_o = np.array([0,15,1,16,2,17,3,18,4,19,5,20,6,21,7,22,8,23,9,24,10,25,11,26,12,27,13,28,14,29])
    new_idx = (np.tile(idx, (numbouts.shape[1],1)).T, np.tile(t_o, (numbouts.shape[0],1)))

    numbouts = numbouts[new_idx]

    f, ax = plt.subplots()
    
    sns.heatmap(numbouts, vmin=0, vmax=np.nanmax(numbouts), cmap='coolwarm', mask=np.isnan(numbouts))
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Fish number')
    #ax.set_title('Ranked #Bouts variation across trials for each fish')

    f.savefig(data_path / 'trial_numbouts_ranked.pdf')
    f.savefig(data_path / 'trial_numbouts_ranked.png')

    plt.close(f)

    return

def corrTFB(experiment, pre, post):

    data_path = path.Path() / '..' / experiment
    tmp = hdf.loadmat(data_path / f'data_tfb')

    tfb = tmp['tfb']

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    b_rates = tmp['bout_rate']

    valid = ~np.isnan(b_rates) & ~np.isnan(tfb) & (tfb <= 2) & (b_rates < 150)
    stats = spearmanr(b_rates[valid].ravel(), tfb[valid].ravel()) # 15 bins, each size 10

    f, ax = plt.subplots()

    ax.scatter(b_rates[valid].ravel(), tfb[valid].ravel(), label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}', alpha=0.05, color=coloring)
    ax.set_xlabel('Baseline bout rate')
    ax.set_ylabel('Time to first bout after stimulus onset')
    #ax.set_title('Correlation between Baseline bout rate and TFB')
    ax.grid(False)
    ax.legend()
    sns.despine(top=True, right=True)

    f.savefig(data_path / 'tfb_rate.pdf')
    f.savefig(data_path / 'tfb_rate.png')

    plt.close(f)

    return

def rateNumBoutCorr(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    numbouts = tmp['incorrect_data']
    b_rates = tmp['bout_rate']

    valid = ~np.isnan(b_rates) & ~np.isnan(numbouts)
    stats = spearmanr(b_rates[valid].ravel(), numbouts[valid].ravel())

    f, ax = plt.subplots()

    ax.scatter(b_rates.ravel(), numbouts.ravel(), alpha=0.05, label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}', color=coloring)
    ax.set_xlabel('Baseline bout rate')
    ax.set_ylabel('#Bouts used for performance')
    #ax.set_title('Correlation between baseline bout rate and #Bouts')
    ax.grid(False)
    ax.legend()
    sns.despine(top=True, right=True)

    f.savefig(data_path / 'rate_numbouts_corr.pdf')
    f.savefig(data_path / 'rate_numbouts_corr.png')

    plt.close(f)

    return

def perfNumBoutCorr(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    numbouts = tmp['incorrect_data']
    performance = tmp['performance']

    valid = ~np.isnan(numbouts) & ~np.isnan(performance)
    stats = spearmanr(numbouts[valid].ravel(), performance[valid].ravel())

    f, ax = plt.subplots()

    ax.scatter(numbouts.ravel(), performance.ravel(), alpha=0.05, label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}')
    ax.set_ylabel('Performance')
    ax.set_xlabel('#Bouts used for performance')
    #ax.set_title('Correlation between #Bouts and performance')
    ax.grid(False)
    ax.legend()
    ax.set_ylim(-1.1,1.1)
    sns.despine(top=True, right=True)

    f.savefig(data_path / 'perf_numbouts_corr.pdf')
    f.savefig(data_path / 'perf_numbouts_corr.png')

    plt.close(f)

    means, _, _ = binned_statistic(numbouts[valid].ravel(), performance[valid].ravel(), statistic=np.nanmean, range=(0,60), bins=12)
    sems, _, _ = binned_statistic(numbouts[valid].ravel(), performance[valid].ravel(), statistic=lambda x: sem(x, nan_policy='omit'), range=(0,60), bins=12)
    x = np.arange(0,60,5)

    f, ax = plt.subplots()

    ax.errorbar(x+2.5, means, yerr=sems, marker='o', capsize=2.0, ecolor='gray', color=coloring)
    ax.set_ylabel('Performance')
    ax.set_xlabel('#Bouts used for performance')
    #ax.set_title('Correlation between #Bouts and performance')
    ax.grid(False)
    ax.legend()
    ax.set_ylim(-1.1,1.1)
    sns.despine(top=True, right=True)

    f.savefig(data_path / 'perf_numbouts_corr_binned.pdf')
    f.savefig(data_path / 'perf_numbouts_corr_binned.png')

    plt.close(f)

    return

def stimNumBoutCorr(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    numbouts = tmp['incorrect_data']
    s_rates = tmp['stim_rate']

    valid = ~np.isnan(numbouts) & ~np.isnan(s_rates)
    stats = spearmanr(numbouts[valid].ravel(), s_rates[valid].ravel())

    f, ax = plt.subplots()

    ax.scatter(s_rates.ravel(), numbouts.ravel(), alpha=0.05, label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}', color=coloring)
    ax.set_xlabel('Stimulus bout rate')
    ax.set_ylabel('#Bouts used for performance')
    #ax.set_title('Correlation between stimulus bout rate and #Bouts')
    ax.grid(False)
    ax.legend()
    sns.despine(top=True, right=True)

    f.savefig(data_path / 'stim_numbouts_corr.pdf')
    f.savefig(data_path / 'stim_numbouts_corr.png')

    plt.close(f)

    return

def shufflePerf(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    performance = tmp['performance']

    performance = performance.reshape(performance.shape[0], -1)
    dims = performance.shape

    result = np.zeros(dims[0]*dims[1])
    for i in range(1):
        result += np.random.permutation(performance.ravel())
    performance = (result / 1).reshape(dims)

    avg = np.nanmean(performance, axis=1)
    idx = np.flip(np.argsort(avg))
    t_o = np.array([0,15,1,16,2,17,3,18,4,19,5,20,6,21,7,22,8,23,9,24,10,25,11,26,12,27,13,28,14,29])
    new_idx = (np.tile(idx, (dims[1],1)).T, np.tile(t_o, (dims[0],1)))

    performance = performance[new_idx]

    f, ax = plt.subplots()
    
    sns.heatmap(performance, vmin=-1.1, vmax=1.0, cmap='coolwarm' ,center=0, mask=np.isnan(performance))
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Fish number')
    #ax.set_title('Shuffled + ranked performance variation')

    f.savefig(data_path / 'trial_performance_shuffled_ranked.pdf')
    f.savefig(data_path / 'trial_performance_shuffled_ranked.png')

    plt.close(f)

    return

def cutCorr(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    raw = tmp['raw_cut']
    bout = tmp['bout_cut']

    valid = ~np.isnan(raw) & ~np.isnan(bout)
    stats = spearmanr(raw[valid].ravel(), bout[valid].ravel())

    f, ax = plt.subplots()

    ax.scatter(raw.ravel(), bout.ravel(), alpha=0.05, label=f'Spearman r:{stats[0]:.2f}, p:{stats[1]:.2e}')
    ax.set_xlabel('Raw proportion')
    ax.set_ylabel('Bout proportion')
    #ax.set_title('Correlation between raw time and bout based filtering')
    ax.grid(False)
    ax.legend()
    sns.despine(top=True, right=True)

    f.savefig(data_path / 'cut_corr.pdf')
    f.savefig(data_path / 'cut_corr.png')

    plt.close(f)

    return

def checkNan(experiment, pre, post):

    data_path = path.Path() / '..' / experiment
    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')

    p = tmp['performance']
    i = tmp['incorrect_data']
    b = tmp['bout_rate']
    s = tmp['stim_rate']

    print(np.isnan(p).sum())
    print(np.isnan(i).sum())
    print(np.isnan(b).sum())
    print(np.isnan(s).sum())
    print(f'{np.isnan(p).sum() / p.ravel().shape[0]: .4f}')

    return

def checkCut(experiment, pre, post):

    data_path = path.Path() / '..' / experiment
    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')

    cut = tmp['raw_cut']
    total = cut.ravel().shape[0]

    for i in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        prop = (cut < i).sum() / total
        print(f'{i} : {prop:.2f}')

    return

def vecBoutPerf(experiment, pre, post):

    data_path = path.Path() / '..' / experiment
    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')

    numbouts = tmp['incorrect_data'].ravel()
    performance = tmp['performance'].ravel()

    idx = np.flip(np.argsort(performance))
    idx2 = np.flip(np.argsort(numbouts))

    f, ax = plt.subplots(figsize=(15,8))
    ax2 = plt.twinx(ax)

    ax.plot(range(idx.shape[0]), performance[idx], marker='o', markersize='2', label='performance', color=coloring)
    ax2.scatter(range(idx.shape[0]), numbouts[idx], marker='o', s=2, color='red', label='numbouts')

    ax.set_xlabel('Sorted idx')
    ax.set_ylabel('Performance')
    ax2.set_ylabel('Num Bouts')
    #ax.set_title('#Bouts varying with ranked performance')
    #ax.legend()
    ax2.legend()
    ax.grid(False)
    ax2.grid(False)
    ax.set_ylim(-1.1,1.1)
    #sns.despine(top=True)

    plt.show()

    f.savefig(data_path / 'vector_numbout_perf.pdf')
    f.savefig(data_path / 'vector_numbout_perf.png')

    plt.close(f)

    return

def createData(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    performance = tmp['performance']
    base = tmp['bout_rate']

    performance = performance.reshape(performance.shape[0], -1)
    base = base.reshape(base.shape[0], -1)

    t_o = np.array([0,15,1,16,2,17,3,18,4,19,5,20,6,21,7,22,8,23, \
        9,24,10,25,11,26,12,27,13,28,14,29])
    new_idx = (np.tile(np.arange(performance.shape[0]), (performance.shape[1],1)).T, np.tile(t_o, (performance.shape[0],1)))

    performance = performance[new_idx]
    base = base[new_idx]
    print(performance.shape, base.shape)
    hdf.savemat('sample_data', {'baseline_rate': base, 'performance': performance}, format='7.3', oned_as='column', store_python_metadata=True)

    return

def printVals(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    performance = np.nanmean(tmp['performance'])
    base = np.nanmean(tmp['bout_rate'])

    sp = sem(tmp['performance'].ravel(), nan_policy='omit')
    sb = sem(tmp['bout_rate'].ravel(), nan_policy='omit')


    lost = np.nansum(tmp['performance'] < 0.75) / (~np.isnan(tmp['performance'])).sum()
    attend = np.nansum(tmp['performance'] >= 0.75) / (~np.isnan(tmp['performance'])).sum()

    print(f'Attentive: {attend: .2f}')
    print(f'Inattentive: {lost: .2f}')

    return

def extremeNumBouts(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    performance = tmp['performance']
    numbouts = tmp['incorrect_data']

    extremes = np.abs(performance) == 1
    filt_numbouts = numbouts[extremes]
    
    v, b = np.histogram(filt_numbouts, range=(0,60), bins=60)
    b = 0.5*(b[1:] + b[:-1])    

    f, ax = plt.subplots()

    ax.plot(b, v, marker='o', markersize=3, color=coloring)

    ax.set_xlabel('Number of bouts')
    ax.set_ylabel('Count')
    #ax.set_title(f'Bouts used for performance of $\pm$ 1.0')
    ax.grid(False)
    sns.despine(top=True, right=True)

    f.savefig(data_path / f'extreme_bouts.pdf')
    f.savefig(data_path / f'extreme_bouts.png')

    plt.close(f)

    return

def numBoutDist(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    numbouts = tmp['incorrect_data']

    v, b = np.histogram(numbouts, range=(0,60), bins=60)
    b = 0.5*(b[1:] + b[:-1])
    v = v/v.sum()

    f, ax = plt.subplots()

    ax.plot(b, v, marker='o', markersize=3, color=coloring)
    ax.set_xlabel('Number of Bouts')
    ax.set_ylabel('Normalized count')
    #ax.set_title('Distribution of number of bouts')
    ax.set_ylim(0,0.15)
    ax.grid(False)
    sns.despine(top=True, right=True)

    f.savefig(data_path / f'numbout_dist.pdf')
    f.savefig(data_path / f'numbout_dist.png')

    plt.close(f)

    return

def binomialPerf(experiment, pre, post):

    data_path = path.Path() / '..' / experiment

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    numbouts = tmp['incorrect_data']
    performance = tmp['performance']

    filt = ~np.isnan(numbouts) & (numbouts > 0)

    possible_vals = numbouts[filt].ravel()

    score_vals = []
    score_bins = []

    f, ax = plt.subplots()

    for val in possible_vals:
        vals = binom.pmf(np.arange(val+1), val, 0.5)
        bins = np.arange(val+1) / val

        score_vals.extend(vals)
        score_bins.extend(bins)

        ax.plot(2*bins-1, vals, alpha=0.005, color='red', lw=1)

    score_vals = np.array(score_vals)
    score_bins = np.array(score_bins)

    v, b, _ = binned_statistic(score_bins, score_vals, statistic=np.nanmean, range=(0,1), bins=20)
    b = 0.5*(b[1:] + b[:-1])
    v = v / v.sum()

    ax.plot(2*b - 1, v, label='Theory', color='red', marker='o', markersize=4)
    ax.set_xlabel('Performance')
    ax.set_ylabel('Probability')
    #ax.set_title('Averaged Probability Distribution By NumBouts')
    ax.grid(False)

    v, b = np.histogram(performance, range=(-1,1), bins=20)
    b = 0.5*(b[1:] + b[:-1])
    v = v / v.sum()
    ax.plot(b, v, marker='o', markersize=4, label='Expt', color='black')
    ax.legend()

    f.savefig(data_path / f'binomial_compare.pdf')
    f.savefig(data_path / f'binomial_compare.png')

    plt.close(f)

    return

#### This is a hack #################
modeling = True
coloring = 'red' if modeling else 'black' 
######################################

if __name__ == '__main__':

    pre = 120; post = 30
    
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
    
    experiments = ['experimental_analysis']
    # experiments = ['simulation']
    
    for experiment in experiments:

        # groupTFB(experiment, pre, post) # Not for model
        # corrTFB(experiment, pre, post) # Not for model
        explorePerf(experiment, pre, post)
        perfOrder(experiment, pre, post)
        individualPerf(experiment, pre, post)
        rateOrder(experiment, pre, post)
        rankedRate(experiment, pre, post)
        rankedPerf(experiment, pre, post)
        numBoutsOrder(experiment, pre, post)
        rankedNumBouts(experiment, pre, post)
        rateNumBoutCorr(experiment, pre, post)
        perfNumBoutCorr(experiment, pre, post)
        stimNumBoutCorr(experiment, pre, post)
        rankedStim(experiment, pre, post)
        shufflePerf(experiment, pre, post)
        printVals(experiment, pre, post)
        extremeNumBouts(experiment, pre, post) 
        binomialPerf(experiment, pre, post)
        numBoutDist(experiment, pre, post)