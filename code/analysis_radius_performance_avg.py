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

    pos_x = raw_data[start]['fish_position_x']
    pos_y = raw_data[start]['fish_position_y']
    bout_stamps = raw_data[start]['timestamp']

    lim1, lim2 = 150, 150+post # Now for OMR performance
    
    omr_bouts = (bout_stamps > lim1) & (bout_stamps < lim2) 
    radius = (pos_x**2 + pos_y**2)[omr_bouts]

    if omr_bouts.sum() == 0:
        performance = np.nan
        radius = np.nan
    else:
        angles = raw_data[start]['fish_accumulated_orientation'] \
            - raw_data[end]['fish_accumulated_orientation']

        valid_angles = angles[omr_bouts]

        if stimulus == 0:
            performance = (valid_angles < 0).sum() + 0.5*(valid_angles==0).sum()
        else:
            performance = (valid_angles > 0).sum() + 0.5*(valid_angles==0).sum()
        
        performance = np.around(performance / valid_angles.shape[0], 2)

    return np.nanmean(np.sqrt(radius)), 2*performance - 1

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

    data_radius = np.full((total_fish, stimuli, trials), np.nan)
    data_performance = np.full((total_fish, stimuli, trials), np.nan)

    
    for day_idx, day in enumerate(days):

        for f in range(fish[day_idx]):

            for t in range(trials):
                
                folder = root / folders[day_idx] / f'{day}_fish{f+1:03d}' \
                    / 'raw_data' / f'trial{t:03d}.dat'
                tmp = open(folder, 'rb')
                raw_data = pickle.load(tmp)

                for stimulus in range(stimuli):    
                    radius, performance = findBouts(raw_data, stimulus, pre, post)
                    data_radius[fish_ctr, stimulus, t] = radius
                    data_performance[fish_ctr, stimulus, t] = performance

            fish_ctr += 1
                    
        print(day, fish_ctr, 'fish done')
        
    return data_radius, data_performance

def processData(experiment, data_radius, data_performance):

    to_save = {}

    to_save['radius'] = data_radius
    to_save['performance'] = data_performance

    return to_save

def main(experiment, pre, post):

    root = path.Path() / '..' / '..'
    
    data_radius, data_performance = extractData(experiment, root, pre, post)

    to_save = processData(experiment, data_radius, data_performance)

    save_dir = path.Path() / '..' / experiment / f'data_avg_radius_pre_{pre}_post_{post}'
    hdf.savemat(save_dir, to_save, format='7.3', oned_as='column', store_python_metadata=True)

    return 0

def plotData(data, pre, post):

    data_path = path.Path() / '..' / 'data'
    tmp = hdf.loadmat(data_path / data)
    
    save_dir = path.Path() / '..' / 'results'
    os.makedirs(save_dir, exist_ok=True)

    radius = tmp['radius']
    performance = tmp['performance']

    stats = spearmanr(radius.ravel(), performance.ravel(), nan_policy='omit')

    f, ax = plt.subplots()

    ax.set_xscale('function', functions=(lambda x: x**2, lambda x: x**0.5))
    ax.scatter(radius.ravel(), performance.ravel(), alpha=0.05, label=f'Spearman r: {stats[0]:.2f}', color='black')
    ax.set_xlabel('Avg. radius')
    ax.set_ylabel('Avg. Performance')
    #ax.set_title('Correlation of performance with radius')
    ax.grid(False)
    #ax.legend()
    ax.set_ylim(-1.1,1.1)
    ax.set_xlim(0,1.01)

    f.savefig(save_dir / 'radius_perf_avg_corr.pdf')
    f.savefig(save_dir / 'radius_perf_avg_corr.png')
    # plt.show()
    plt.close(f)

    return 0

def plotRadBase(data, pre, post):

    data_path = path.Path() / '..' / 'data'
    tmp = hdf.loadmat(data_path / data)
    
    save_dir = path.Path() / '..' / 'results'
    os.makedirs(save_dir, exist_ok=True)

    radius = tmp['radius']

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_{pre}_post_{post}')
    base = tmp['stim_rate']

    valid = ~np.isnan(base)

    stats = spearmanr(radius[valid].ravel(), base[valid].ravel(), nan_policy='omit')

    f, ax = plt.subplots()

    ax.scatter(base[valid].ravel(), radius[valid].ravel(), alpha=0.05, label=f'Spearman r: {stats[0]:.2f}', color='black')
    ax.set_ylabel('Avg. radius')
    ax.set_xlabel('Stimulus bout rate')
    #ax.set_title('Correlation of stimulus bout rate with radius')
    ax.grid(False)
    #ax.legend()
    ax.set_xlim(0,150)
    ax.set_ylim(0,1.1)

    f.savefig(save_dir / 'radius_stim_avg_corr.pdf')
    f.savefig(save_dir / 'radius_stim_avg_corr.png')

    plt.close(f)

    return 0

def plotRadDist(data, pre, post):

    data_path = path.Path() / '..' / 'data'
    tmp = hdf.loadmat(data_path / data)
    
    save_dir = path.Path() / '..' / 'results'
    os.makedirs(save_dir, exist_ok=True)

    radius = tmp['radius']

    v, b = np.histogram(radius, range=(0,1), bins=10)
    b = 0.5*(b[1:] + b[:-1])
    v = v / v.sum()

    f, ax = plt.subplots()

    ax.plot(b, v, marker='o', markersize=4, color='black')
    ax.set_xlabel('Radius')
    ax.set_ylabel('Counts')
    #ax.set_title('Distribution of avg. radius at bout start during trial')
    ax.grid(False)
    ax.set_ylim(0,1.1)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / 'radius_dist_avg.pdf')
    f.savefig(save_dir / 'radius_dist_avg.png')

    plt.close(f)



    return

if __name__ == '__main__':

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
    plt.rc('figure', figsize=(6,4))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.facecolor'] = (0,0,0,0)
    plt.rcParams['axes.facecolor'] = (0,0,0,0)
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['axes.linewidth'] = 0.25
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = False

    pre = 120; post = 30
    data = f'data_avg_radius_pre_{pre}_post_{post}'
    
    #main(experiment, pre, post)
    plotData(data, pre, post)
    #plotRadBase(data, pre, post)
    # plotRadDist(data, pre, post)

    sys.exit(0)
