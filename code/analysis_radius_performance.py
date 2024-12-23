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
    bout_stamps = raw_data[start]['timestamp']

    lim1, lim2 = 150, 150+post # Now for OMR performance
    #lim1, lim2 = 30, 150
    #lim1 = np.random.randint(150-pre, 120) # This is to find 30 second random window to match stimulus duration
    #lim2 = lim1 + 30
    
    omr_bouts = (bout_stamps > lim1) & (bout_stamps < lim2) 
    radius = (pos_x**2 + pos_y**2)[omr_bouts]

    if omr_bouts.sum() <= 0:
        performance = np.array([])
        radius = np.array([])
    else:
        angles = raw_data[start]['fish_accumulated_orientation'] \
            - raw_data[end]['fish_accumulated_orientation']

        valid_angles = angles[omr_bouts]

        if stimulus == 0:
            performance = (valid_angles < 0) + 0.5*(valid_angles==0)
        else:
            performance = (valid_angles > 0) + 0.5*(valid_angles==0)

    return np.sqrt(radius), performance

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

    data_radius = []
    data_performance = []

    
    for day_idx, day in enumerate(days):

        for f in range(fish[day_idx]):

            for t in range(trials):
                
                folder = root / folders[day_idx] / f'{day}_fish{f+1:03d}' \
                    / 'raw_data' / f'trial{t:03d}.dat'
                tmp = open(folder, 'rb')
                raw_data = pickle.load(tmp)

                for stimulus in range(stimuli):    
                    radius, performance = findBouts(raw_data, stimulus, pre, post)
                    #data_radius[fish_ctr, stimulus, t] = radius
                    #data_performance[fish_ctr, stimulus, t] = performance
                    data_radius.extend(radius.tolist())
                    data_performance.extend(performance.tolist())

            fish_ctr += 1
                    
        print(day, fish_ctr, 'fish done')
        
    return np.array(data_radius), 2*np.array(data_performance) - 1

def processData(experiment, data_radius, data_performance):

    to_save = {}

    to_save['radius'] = data_radius
    to_save['performance'] = data_performance

    return to_save

def main(experiment, pre, post):

    root = path.Path() / '..' / '..'
    
    data_radius, data_performance = extractData(experiment, root, pre, post)

    to_save = processData(experiment, data_radius, data_performance)

    save_dir = path.Path() / '..' / experiment / f'data_radius_pre_{pre}_post_{post}'
    hdf.savemat(save_dir, to_save, format='7.3', oned_as='column', store_python_metadata=True)

    return 0

def plotData(data, pre, post):

    data_path = path.Path() / '..' / 'data'
    tmp = hdf.loadmat(data_path / data)
    
    save_dir = path.Path() / '..' / 'results'
    os.makedirs(save_dir, exist_ok=True)

    radius = tmp['radius']
    performance = tmp['performance']

    means, _, _ = binned_statistic(radius, performance, statistic=np.nanmean, range=(0,1), bins=10)
    sems, _, _ = binned_statistic(radius, performance, statistic=lambda x: sem(x, nan_policy='omit'), range=(0,1), bins=10)

    x = np.linspace(0,1,10)

    sns.set_style('white')
    sns.set_style('ticks')

    f, ax = plt.subplots()

    ax.errorbar(x, means, yerr=sems, capsize=2.0, markersize=4, ecolor='gray', color='black')
    ax.set_xlabel('Radius')
    ax.set_ylabel('Avg. Performance')
    ax.set_title('Performance variation with radius')
    ax.grid(False)
    ax.set_ylim(-1.1,1.1)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / 'radius_perf.pdf')
    f.savefig(save_dir / 'radius_perf.png')

    plt.close(f)

    return 0

def radDist(data, pre, post):

    data_path = path.Path() / '..' / 'data'
    tmp = hdf.loadmat(data_path / data)
    
    save_dir = path.Path() / '..' / 'results'
    os.makedirs(save_dir, exist_ok=True)

    radius = tmp['radius']

    v, b = np.histogram(radius, range=(0,1), bins=10)
    #v = v / (b[1:]**2)
    v = v / v.sum()
    b = 0.5*(b[1:] + b[:-1])

    sns.set_style('white')
    sns.set_style('ticks')

    f, ax = plt.subplots()

    ax.plot(b, v, marker='o', markersize=4, color='black')
    ax.set_xlabel('Radius')
    ax.set_ylabel('Counts')
    ax.set_title('Distribution of radius at bout start')
    ax.grid(False)
    ax.set_ylim(0,1.1)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / 'radius_dist.pdf')
    f.savefig(save_dir / 'radius_dist.png')

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

    data = f'data_radius_pre_{pre}_post_{post}'

    #main(experiment, pre, post)
    plotData(data, pre, post)
    radDist(data, pre, post)

    sys.exit(0)
