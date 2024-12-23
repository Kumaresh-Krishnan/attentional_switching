import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import hdf5storage as hdf
import path
import os
from scipy.stats import sem, binned_statistic

def psychometrics():

    dp1 = path.Path() / '..' / 'data'
    dp2 = path.Path() / '..' / 'data'
    dp3 = path.Path() / '..' / 'data'

    tmp1 = hdf.loadmat(dp1 / f'baseline_data_bout_rate_pre_{120}_post_{30}')
    tmp2 = hdf.loadmat(dp2 / f'ctrl_50_data_bout_rate_pre_{120}_post_{30}')
    tmp3 = hdf.loadmat(dp3 / f'expt_50_data_bout_rate_pre_{120}_post_{30}')

    c1 = tmp1['performance']
    c2 = tmp2['performance']
    c3 = tmp3['performance']

    p_n100 = np.nanmean(c1[:,0,:])
    p_n50 = np.nanmean(c2[:,0,:])
    p_100 = np.nanmean(c1[:,1,:])
    p_50 = np.nanmean(c2[:,1,:])
    p_0 = np.nanmean(c3)

    acc = [-p_n100, -p_n50, p_0, p_50, p_100]
    print(acc)
    xvals = [-100, -50, 0, 50, 100]

    f, ax = plt.subplots()

    ax.plot(xvals, acc, color='black', marker='o')
    ax.axhline(0, linestyle='--', color='black', alpha=0.3)
    ax.axvline(0, linestyle='--', color='black', alpha=0.3)

    ax.set_ylim(-1.1,1.1)
    ax.set_xlim(-105,105)
    ax.set_xticks([-100,-50,0,50,100])
    ax.set_xlabel('Coherence level (%)')
    ax.set_ylabel('Performance - Leftward to Rightward')

    #f.savefig(f'psycho_expt_full.pdf')
    f.savefig(f'../results/psycho_expt_full.png')

    plt.close(f)

    return

def turn():

    dp1 = path.Path() / '..' / 'data' / 'multitrial_50_ctrl'
    dp2 = path.Path() / '..' / 'data' / 'multitrial_50_expt'
    tmp1 = hdf.loadmat(dp1 / f'ctrl_50_data_angles_{150}_{180}_361')
    tmp2 = hdf.loadmat(dp2 / f'expt_50_data_angles_{150}_{180}_361')
    tmp3 = hdf.loadmat(dp1 / f'ctrl_50_data_angles_{30}_{150}_361')
    
    id_map = {'0':'Leftward', '1':'Rightward'}
    angles = np.linspace(-180,180,72)

    m1 = tmp1['mean_prob']
    s1 = tmp1['sem_prob']
    m2 = tmp2['mean_prob']
    s2 = tmp2['sem_prob']
    m3 = tmp3['mean_prob']
    s3 = tmp3['sem_prob']


    for stimulus in range(2): # We know it is 2 stimuli, hard coding for paper figure

        f, ax = plt.subplots()

        ax.plot(angles, m1[stimulus], color='black', marker='None', markersize=2)
        ax.plot(angles, m2[stimulus], color='black', alpha=0.8, marker='None', markersize=2, linestyle='dotted')
        ax.plot(angles, m3[stimulus], color='black', linestyle='--', alpha=0.6, marker='None', markersize=2)

        ax.fill_between(angles, m1[stimulus]-s1[stimulus], m1[stimulus]+s1[stimulus], color='grey', alpha=0.2)
        ax.fill_between(angles, m2[stimulus]-s2[stimulus], m2[stimulus]+s2[stimulus], color='grey', alpha=0.2)
        ax.fill_between(angles, m3[stimulus]-s3[stimulus], m3[stimulus]+s3[stimulus], color='grey', alpha=0.2)

        ax.set_xlabel(f'Turn Angle ($^\circ$)')
        ax.set_ylabel('Probability')
        ax.set_ylim(0,0.3)
        ax.set_xticks(np.arange(-100,101,50))

        f.savefig(f'turn_{id_map[str(stimulus)]}.pdf')
        f.savefig(f'turn_{id_map[str(stimulus)]}.png')

        plt.close(f)

    return

def tfb():

    dp1 = path.Path() / '..' / 'data' 
    dp2 = path.Path() / '..' / 'data' 

    tmp1 = hdf.loadmat(dp1 / f'data_tfb')
    tmp2 = hdf.loadmat(dp2 / f'baseline_data_tfb')

    tf1 = tmp1['tfb']
    tf2 = tmp2['tfb']

    tp1 = 2*tmp1['tfb_performance']-1
    tp2 = tmp2['tfb_performance'] # This data was already saved in [-1,1] sheesh!

    valids = ~np.isnan(tf1) # Remove nans from the data
    tf1 = tf1[valids]
    tp1 = tp1[valids]
    valids = ~np.isnan(tf2) # Remove nans from the data
    tf2 = tf2[valids]
    tp2 = tp2[valids]

    v1, b1, _ = binned_statistic(tf1, tp1, statistic=np.nanmean, range=(0,2), bins=20)
    e1, b1, _ = binned_statistic(tf1, tp1, statistic=lambda x: sem(x, nan_policy='omit'), range=(0,2), bins=20)
    v2, b2, _ = binned_statistic(tf2, tp2, statistic=np.nanmean, range=(0,2), bins=20)
    e2, b2, _ = binned_statistic(tf2, tp2, statistic=lambda x: sem(x, nan_policy='omit'), range=(0,2), bins=20)

    f, ax = plt.subplots()

    ax.errorbar(b1[:-1], v1, yerr=e1, elinewidth=0.25, capsize=1.0, ecolor='gray', color='black', marker='o', markersize=4)
    ax.errorbar(b2[:-1], v2, yerr=e2, elinewidth=0.25, capsize=1.0, ecolor='gray', alpha=0.4, color='black', linestyle='--', \
                marker='^', markersize=4)

    ax.set_xlabel('Time to first bout (s)')
    ax.set_ylabel('Performance')
    ax.set_xlim(0,2.04)
    ax.set_ylim(-1.1,1.1)

    f.savefig('../results/tfb_full.pdf')

    plt.close(f)

    return

def perf():

    dp1 = path.Path() / '..' / 'data'
    dp2 = path.Path() / '..' / 'data'

    tmp1 = hdf.loadmat(dp1 / f'data_bout_rate_pre_{120}_post_{30}')
    tmp2 = hdf.loadmat(dp2 / f'baseline_data_bout_rate_pre_{120}_post_{30}')

    c1 = tmp1['performance']
    c2 = tmp2['performance']

    v1, b1 = np.histogram(c1, range=(-1,1), bins=21)
    b1 = (b1[1:] + b1[:-1])*0.5
    v1 = v1 / v1.sum()

    v2, b2 = np.histogram(c2, range=(-1,1), bins=21)
    b2 = (b2[1:] + b2[:-1])*0.5
    v2 = v2 / v2.sum()

    f, ax = plt.subplots()
    
    ax.plot(b1, v1, color='black', marker='o', markersize=4)
    ax.plot(b2, v2, color='black', alpha=0.4, linestyle='--', marker='^', markersize=4)

    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(0,0.6)
    ax.set_xlabel('Performance')
    ax.set_ylabel('Probability')

    f.savefig('../results/perf_reduced.pdf')

    plt.close(f)

    return

def radPerf():

    dp1 = path.Path() / '..' / 'data'
    dp2 = path.Path() / '..' / 'data'

    tmp1 = hdf.loadmat(dp1 / f'data_radius_pre_{120}_post_{30}')
    tmp2 = hdf.loadmat(dp2 / f'baseline_data_radius_pre_{120}_post_{30}')

    r1 = tmp1['radius']
    p1 = tmp1['performance']
    r2 = tmp2['radius']
    p2 = tmp2['performance']

    m1, _, _ = binned_statistic(r1, p1, statistic=np.nanmean, range=(0,1), bins=10)
    s1, _, _ = binned_statistic(r1, p1, statistic=lambda x: sem(x, nan_policy='omit'), range=(0,1), bins=10)
    m2, _, _ = binned_statistic(r2, p2, statistic=np.nanmean, range=(0,1), bins=10)
    s2, _, _ = binned_statistic(r2, p2, statistic=lambda x: sem(x, nan_policy='omit'), range=(0,1), bins=10)

    x = np.linspace(0,1,10)

    f, ax = plt.subplots()

    ax.errorbar(x, m1, yerr=s1, capsize=5.0, markersize=4, ecolor='gray', color='black', elinewidth=0.25, marker='o')
    ax.errorbar(x, m2, yerr=s2, capsize=5.0, markersize=4, ecolor='gray', color='black', elinewidth=0.25, linestyle='--', alpha=0.4, \
                marker='^')
    ax.set_xlabel('Radius')
    ax.set_ylabel('Avg. Performance')
    ax.set_ylim(-1.1,1.1)


    f.savefig('../results/radius_perf_full.pdf')
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

    #turn()
    #psychometrics()
    # tfb()
    perf()
    #radPerf()
