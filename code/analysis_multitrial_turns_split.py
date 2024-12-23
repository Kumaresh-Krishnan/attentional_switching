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
from scipy.stats import binned_statistic, sem

import path
import pickle

def findBouts(raw_data, stimulus, num_bins, s, e):

    start = f'bouts_start_stimulus_{stimulus:03d}'
    end = f'bouts_end_stimulus_{stimulus:03d}'

    errors = raw_data[f'raw_stimulus_{stimulus:03d}']['errorcode']
    raw_time = raw_data[f'raw_stimulus_{stimulus:03d}']['timestamp']

    pos_x = raw_data[start]['fish_position_x']
    pos_y = raw_data[start]['fish_position_y']
    
    bout_stamps = raw_data[start]['timestamp']
    
    lim1, lim2 = s, e # Now for OMR performance
    #lim1 = np.random.randint(30, 120) # This is to find 30 second random window to match stimulus duration
    #lim2 = lim1 + 30
    
    omr_bouts = (bout_stamps > lim1) & (bout_stamps < lim2) & (pos_x**2 + pos_y**2 < 0.81)

    if omr_bouts.sum() <= 0:
        angle_rates = np.array([np.nan]*num_bins)
        correct = np.nan
    else:
        angles = raw_data[start]['fish_accumulated_orientation'] \
            - raw_data[end]['fish_accumulated_orientation']

        valid_angles = angles[omr_bouts]

        counts, _ = np.histogram(valid_angles, range=(-180,180), bins=num_bins)
        rpos_x = raw_data[f'raw_stimulus_{stimulus:03d}']['fish_position_x']
        rpos_y = raw_data[f'raw_stimulus_{stimulus:03d}']['fish_position_y']

        window = (raw_time > lim1) & (raw_time < lim2) 
        window_cut = window & (rpos_x**2 + rpos_y**2 < 0.81)
        cut = window_cut.sum() / window.sum()
        window_errors = errors[window_cut]
        valid = (window_errors == 0).sum() / window_errors.shape[0]
        
        angle_rates = counts / ((lim2-lim1)*valid*cut)

        if stimulus == 0:
            correct = (valid_angles < 0).sum() + 0.5*(valid_angles == 0).sum()
        else:
            correct = (valid_angles > 0).sum() + 0.5*(valid_angles == 0).sum()

        correct = correct / valid_angles.shape[0]

        if correct > 0.86: # Change this based on attentive or inattentive
            angle_rates = np.array([np.nan]*num_bins)
            correct = np.nan
        
    return angle_rates, correct

def extractData(experiment, root, start, end):

    info_path = path.Path() / '..' / experiment

    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()
    days = info['days']
    fish = info['fish']
    trials = info['trials']
    folders = info['folders']
    stimuli = info['stimuli']
    total_fish = np.sum(fish)

    fish_ctr = 0
    num_bins = 72 # Use 361 for creating distributions

    data_angles = np.zeros((stimuli, total_fish, trials, num_bins))
    data_correct = np.zeros((stimuli, total_fish, trials))
    
    for day_idx, day in enumerate(days):

        for f in range(fish[day_idx]):

            for t in range(trials):
                
                folder = root / folders[day_idx] / f'{day}_fish{f+1:03d}' \
                    / 'raw_data' / f'trial{t:03d}.dat'
                tmp = open(folder, 'rb')
                raw_data = pickle.load(tmp)

                for stimulus in range(stimuli):    
                    angles, correct = findBouts(raw_data, stimulus, num_bins, start, end)
                    data_angles[stimulus, fish_ctr, t] = angles
                    data_correct[stimulus, fish_ctr, t] = correct

            fish_ctr += 1
                    
        print(day, fish_ctr, 'fish done')
        
    return data_angles, data_correct

def processData(experiment, data_angles, data_correct):

    mean_angles = np.nanmean(data_angles, axis=(1,2))

    dims = data_angles.shape
    sem_angles = sem(data_angles.reshape(dims[0],-1,dims[3]), axis=1, nan_policy='omit')

    normalizer = mean_angles.sum(axis=1).reshape(-1,1)

    mean_prob = mean_angles / normalizer
    sem_prob = sem_angles / normalizer

    mean_correct = np.nanmean(data_correct, axis=(1,2))
    
    to_save = {}

    to_save['raw_angles'] = data_angles
    to_save['mean'] = mean_angles
    to_save['sem'] = sem_angles.data if np.ma.isMaskedArray(sem_angles) else sem_angles
    to_save['mean_prob'] = mean_prob
    to_save['sem_prob'] = sem_prob.data if np.ma.isMaskedArray(sem_prob) else sem_prob
    to_save['correct'] = mean_correct
    
    return to_save

def main(experiment, start, end):

    root = path.Path() / '..'
    
    data_angles, data_correct = extractData(experiment, root, start, end)

    to_save = processData(experiment, data_angles, data_correct)

    save_dir = path.Path() / '..' / experiment / f'inattentive_data_angles_{start}_{end}_361' # Change this based on att/inatt
    hdf.savemat(save_dir, to_save, format='7.3', oned_as='column', store_python_metadata=True)

    return 0

def plotData(data, start, end, prob=False):

    data_path = path.Path() / '..' / 'data'
    tmp = hdf.loadmat(data_path / data)
    id_map = {'0':'Leftward', '1':'Rightward'}

    tmp2 = hdf.loadmat(data_path / '../data' / f'data_angles_{30}_{150}_361')
    data_mean2 = tmp2['mean_prob']
    data_sem2 = tmp2['sem_prob']
    data_mean2 = (data_mean2[1] + np.flip(data_mean2[0])) / 2
    data_correct2 = 2*tmp2['correct'] - 1
    data_correct2 = np.nanmean(data_correct2, axis=0)
    data_sem2 = np.sqrt(data_sem2[1]**2 + np.flip(data_sem2[1])**2) / data_sem2.shape[0]

    if prob:
        data_mean = tmp['mean_prob']
        data_sem = tmp['sem_prob']
        save_dir = path.Path() / '..' / 'results' / f'angle_plots_{start}_{end}_prob_361'
    else:
        data_mean = tmp['mean']*60
        data_sem = tmp['sem']*60
        save_dir = path.Path() / '..' / 'results' / f'angle_plots_{start}_{end}_361'

    os.makedirs(save_dir, exist_ok=True)

    angles = np.linspace(-180,180,72) # Hard coded assuming 72 is standard

    data_correct = 2*tmp['correct'] - 1

    data_mean = (data_mean[1] + np.flip(data_mean[0])) / 2
    data_correct = np.nanmean(data_correct, axis=0)
    data_sem = np.sqrt(data_sem[1]**2 + np.flip(data_sem[1])**2) / data_sem.shape[0]

    #for stimulus in range(data_mean.shape[0]):
    print(data_mean, angles)
    f, ax = plt.subplots(figsize=(8,6))

    ax.plot(angles, data_mean, label=f'Perf: {data_correct:.2f}', color='black')
    ax.fill_between(angles, data_mean-data_sem, \
        data_mean+data_sem, color='grey', alpha=0.7)
    
    ax.plot(angles, data_mean2, label=f'Perf: {data_correct2:.2f}', color='black', linestyle='--', alpha=0.7)
    ax.fill_between(angles, data_mean2-data_sem2, \
        data_mean2+data_sem2, color='grey', alpha=0.7)

    ax.set_xlabel(f'Angle turned ($^\circ$)')
    ax.set_ylabel('Bout rate (proportion)') if prob else ax.set_ylabel('Bout rate (bouts/min)')
    ax.set_title(f'Turn angle distribution')
    ax.legend()
    ax.grid(False)
    ax.set_ylim(0,0.2) if prob else ax.set_ylim(0,12)
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'inattentive_doubled_distribution.pdf') # Change based on att/inatt
    f.savefig(save_dir / f'inattentive_doubled_distribution.png') # Change based on att/inatt
    plt.close(f)

    return 0


if __name__ == '__main__':

    experiment = 'multitrial_2'
    pairs = [(150,160), (150,170), (150,180), (160,170), (160,180), (170,180)]
    pairs = [ (150,180)]

    sns.set_style('white')
    sns.set_style('ticks')

    for s,e in pairs:

        # main(experiment, s, e)
        # plotData(experiment, s, e)
        plotData(experiment, s, e, True)

    sys.exit(0)
