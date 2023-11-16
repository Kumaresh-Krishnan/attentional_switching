#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  simulator.py
#  
#  Copyright 2019 Kumaresh <kumaresh_krishnan@g.harvard.edu>
#  
#  version 1.0

import numpy as np
import os, sys
import hdf5storage as hdf

from numba import njit, prange

from integrator import Integrator
from sampler import Sampler

import matplotlib.pyplot as plt
import seaborn as sns

import path

def extractInfo(r1, r2, d1, d2, d3, d4, d5, state, params):

    thresh = params[2]
    tau_mod = params[1] / 1
    
    x = Integrator(t=180, sig=params[0], tau=tau_mod)
    
    if state == 1:
        x.nC(1, 150, 180) # This is hard coded for experiment currently
        
    x.run()
    # print(x.results().tolist()); input()
    s = Sampler(x.results(), d1=d1, d2=d2, d3=d3, d4=d4, d5=d5, b=thresh, r1=r1, r2=r2)
    s.run()
    
    bout_rate = np.count_nonzero(s.bouts[3000:15000]) / 2 # rate / 120 * 60
    incorrect = np.count_nonzero(s.bouts[15000:])
    stim_rate = incorrect * 2 # rate / 30 * 60
    performance = ((s.bouts[15000:] == 1).sum() + 0.5*(s.bouts[15000:] == 0.5).sum()) / incorrect
    
    return bout_rate, performance, incorrect, stim_rate, tau_mod, s.bouts

def changeState(current, r1, trial, s):

    r = np.random.random()
    
    # p - probability to become inattentive (rp - rate, tp - trial)
    rp = (((r1-0.04) / 2.15) / 1.0) # Map to values between 0 and 0.75
    rp = rp**2

    w = 0.5 # How much to combine rate dependence and trial dependence on p
    tp = (trial*2+s) / 30 # trial / 15 but actually 30 trials no?!
    tp = tp**2
    p = w*rp + (1-w)*tp

    state = current

    if current == 1 and r > 1-p:
        state = 0
    elif r > p:
        state = 1
    else:
        state = current

    return state, p

def generate(experiment, pre, post, params):

    info_path = path.Path()

    info = np.load(info_path / experiment / f'expt_info.npy', allow_pickle=True).item()
    fish = info['fish']
    trials = info['trials']
    stimuli = info['stimuli']

    tmp = hdf.loadmat(path.Path() / experiment / 'bout_dist_data.mat')

    r1s = (tmp['bout_rate'] / 60)*1.2
    r2s = (tmp['stim_rate'] / 60)*1.4
    r1s[np.isnan(r1s)] = 1.0
    r2s[np.isnan(r2s)] = 1.2

    data_bout_rate = np.zeros((fish, stimuli, trials))
    data_performance = np.zeros((fish, stimuli, trials))
    data_incorrect_data = np.zeros((fish, stimuli, trials))
    data_stim_rate = np.zeros((fish, stimuli, trials))
    data_state = np.zeros((fish, stimuli, trials))
    data_p = np.zeros((fish, stimuli, trials))
    data_tau = np.zeros((fish, stimuli, trials))
    data_trace = np.zeros((fish, stimuli, trials, 18000)) # Length of stimulus, hard coded

    dists = hdf.loadmat(info_path / experiment / 'distributions.mat')
    d1 = dists['base']
    d2 = dists['left']
    d3 = dists['right']
    d4 = dists['base_left']
    d5 = dists['base_right']

    for f in prange(fish):
        state = np.random.choice([0,1]) # 1 - engage, 0 - disengage

        for t in prange(trials):
            
            for s in prange(stimuli):
                
                state, p = changeState(state, r1s[f,s,t], t, s)
                #state=1              
                base, perf, incorr, stim, tau, trace = extractInfo(r1s[f,s,t], r2s[f,s,t], d1, d2, d3, d4, d5, state, params)

                data_bout_rate[f,s,t] = base
                data_performance[f,s,t] = perf
                data_incorrect_data[f,s,t] = incorr
                data_stim_rate[f,s,t] = stim
                data_state[f,s,t] = state
                data_p[f,s,t] = p
                data_tau[f,s,t] = tau
                data_trace[f,s,t] = trace

    return data_bout_rate, 2*data_performance - 1, data_incorrect_data, data_stim_rate, data_state, data_p, data_tau, data_trace

def processData(data_bout_rate, data_performance, data_incorrect_data, data_stim_rate, data_state, data_p, data_tau, data_trace):
    
    to_save = {}

    to_save['bout_rate'] = data_bout_rate
    to_save['performance'] = data_performance
    to_save['incorrect_data'] = data_incorrect_data
    to_save['stim_rate'] = data_stim_rate
    to_save['state'] = data_state
    to_save['p'] = data_p
    to_save['tau'] = data_tau
    to_save['trace'] = data_trace
    
    return to_save

def main(experiment, pre, post, params):

    data_bout_rate, data_performance, data_incorrect_data, data_stim_rate, data_state, data_p, data_tau, data_trace = \
        generate(experiment, pre, post, params)

    to_save = processData(data_bout_rate, data_performance, \
        data_incorrect_data, data_stim_rate, data_state, data_p, data_tau, data_trace)

    save_dir = path.Path() / experiment / f'data_bout_rate_pre_{pre}_post_{post}'
    hdf.savemat(save_dir, to_save, format='7.3', oned_as='column', store_python_metadata=True)

    return

if __name__ == '__main__':

    experiment = '../simulation'
    pre = 120; post = 30
    main(experiment, pre, post, (8,2.8,0.8))


    sys.exit()
