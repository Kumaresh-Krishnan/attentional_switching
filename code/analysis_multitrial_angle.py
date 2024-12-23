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
        performance = []
        valid_angles = []
    elif omr_bouts.sum() <= 0:
        performance = []
        valid_angles = []
    else:
        angles = raw_data[start]['fish_accumulated_orientation'] \
            - raw_data[end]['fish_accumulated_orientation']

        valid_angles = angles[omr_bouts]

        if stimulus == 0:
            performance = list((valid_angles < 0) + 0.5*(valid_angles==0))
        else:
            performance = list((valid_angles > 0) + 0.5*(valid_angles==0))
            
    return list(valid_angles), performance

def extractData(experiment, root, pre, post):

    info_path = path.Path() / '..' / experiment

    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()
    days = info['days']
    fish = info['fish']
    trials = info['trials']
    folders = info['folders']
    stimuli = info['stimuli']

    fish_ctr = 0

    data_angles = []
    data_performance = []
    
    for day_idx, day in enumerate(days):

        for f in range(fish[day_idx]):

            for t in range(trials):
                
                folder = root / folders[day_idx] / f'{day}_fish{f+1:03d}' \
                    / 'raw_data' / f'trial{t:03d}.dat'
                tmp = open(folder, 'rb')
                raw_data = pickle.load(tmp)

                for stimulus in range(stimuli):    
                    angle, performance = findBouts(raw_data, stimulus, pre, post)
                    data_angles.extend(angle)
                    data_performance.extend(performance)

            fish_ctr += 1
                    
        print(day, fish_ctr, 'fish done')
        
    return np.array(data_angles), np.array(data_performance)

def processData(experiment, data_angles, data_performance, mutant=False):

    to_save = {}

    to_save['angles'] = data_angles
    to_save['performance'] = data_performance
    
    return to_save

def main(experiment, pre, post, mutant):

    root = path.Path() / '..'
    
    data_angles, data_performance = extractData(experiment, root, pre, post)

    to_save = processData(experiment, data_angles, data_performance, mutant)

    save_dir = path.Path() / '..' / experiment / f'data_angle_perf_pre_{pre}_post_{post}'
    hdf.savemat(save_dir, to_save, format='7.3', oned_as='column', store_python_metadata=True)

    return 0

def plotData(data, pre, post):

    data_path = path.Path() / '..' / 'data'
    tmp = hdf.loadmat(data_path / data)
    
    save_dir = path.Path() / '..' / 'results'
    os.makedirs(save_dir, exist_ok=True)

    angles = tmp['angles']
    performance = tmp['performance']

    means, bins, _ = binned_statistic(angles, performance, statistic=np.nanmean, range=(-100,100), bins=100)
    sems, _, _ = binned_statistic(angles, performance, statistic=lambda x: sem(x, nan_policy='omit'), range=(-100,100), bins=100)

    bins = 0.5*(bins[1:] + bins[:-1])

    f, ax = plt.subplots()

    ax.errorbar(bins, means, yerr=sems, capsize=4, ecolor='grey', color='black', marker='o', label='angle_perf')
    ax.axhline(0.5, linestyle='--', color='black')
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Performance score')
    ax.set_ylim(0,1)
    ax.set_xlim(-110,110)
    ax.legend()
    ax.grid(False)
    sns.despine(top=True, right=True)

    f.savefig(f'{save_dir}/angle_performance_2_pre_{pre}_post_{post}.pdf')
    f.savefig(f'{save_dir}/angle_performance_2_pre_{pre}_post_{post}.png')
    plt.close(f)

    return

if __name__ == '__main__':

    # !!!!!!!!!!!!!!!!! If looking at baseline then change lim1/lim2 accordingly !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # s = 15
    # sns.set_style('white')
    # sns.set_style('ticks')
    # plt.rc('font', size=s)          # controls default text sizes
    # plt.rc('axes', titlesize=s)     # fontsize of the axes title
    # plt.rc('axes', labelsize=s)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=s)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=s)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=s-2)    # legend fontsize
    # plt.rc('figure', titlesize=s)  # fontsize of the figure title
    # plt.rc('figure', figsize=(6,4))
    # plt.rcParams['savefig.dpi'] = 300
    # plt.rcParams['savefig.facecolor'] = (0,0,0,0)
    # plt.rcParams['axes.facecolor'] = (0,0,0,0)
    # plt.rcParams['lines.linewidth'] = 0.5
    # plt.rcParams['axes.linewidth'] = 0.25
    # plt.rcParams['axes.spines.top'] = False
    # plt.rcParams['axes.spines.right'] = False
    # plt.rcParams['axes.grid'] = False
    
    pre = 120; post = 30

    data = f'data_angle_perf_pre_{pre}_post_{post}'
    mutant = False # If mutant, have to load genotype information differently, only for main()


    # main(experiment, pre, post, mutant) # comment for modeling
    plotData(data, pre, post) # comment for modeling
    
    sys.exit(0)
