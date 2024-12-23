import hdf5storage as hdf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import path
from scipy.stats import sem



def dists(g1, g2):

    f, ax = plt.subplots()

    ax.hist(g1)
    ax.hist(g2, alpha=0.5)

    plt.show()

def combineBars(e1, e2, label):

    dp1 = path.Path() / '..' / 'data'
    dp2 = path.Path() / '..' / 'data'

    tmp1 = hdf.loadmat(dp1 / e1)
    tmp2 = hdf.loadmat(dp2 / e2)

    p1 = tmp1['all_params']
    p2 = tmp2['all_params']
    
    mean_1 = tmp1['params']
    mean_2 = tmp2['params']
    err_1 = 1.96 * sem(p1, axis=0, nan_policy='omit')
    #err_1 = tmp1['errs']
    err_2 = 1.96 * sem(p2, axis=0, nan_policy='omit')
    #err_2 = tmp2['errs']

    f, (ax, ax2) = plt.subplots(1,2)

    ax.bar([0,1], [1-mean_1[1], 1-mean_2[1]], width=0.5, capsize=1.0, color=(0.993,0.906,0.144), linewidth=0, bottom=[mean_1[1], mean_2[1]])
    ax.bar([0,1], [mean_1[1], mean_2[1]], yerr=[err_1[1], err_2[1]], width=0.5, color=(0.206,0.371,0.553), alpha=0.7, linewidth=0.0, capsize=1.0)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['$w_{ctrl}$', '$w_{expt}$'])
    ax.set_ylim([0,1])

    ax2.bar([0,1], [1/mean_1[0], 1/mean_2[0]], yerr=[err_1[0], err_2[0]], width=0.5, alpha=0.4, color=(0.206,0.371,0.553), capsize=1.0)
    ax2.set_xticks([0,1])
    ax2.set_xticklabels(['$\kappa_{ctrl}$', '$\kappa_{expt}$'])
    ax2.set_ylim([0,3])

    f.savefig(f'param_{label}.pdf')
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

    e1s = ['multitrial_50_ctrl', 'multitrial_50_ctrl', 'multitrial_sleep_ctrl', 'multitrial_scn_all_wt', 'multitrial_slc_all_wt']
    e2s = ['multitrial_50_expt', 'multitrial_distract', 'multitrial_sleep_deprived', 'multitrial_scn_all_mut', 'multitrial_slc_all_mut']
    labels = ['50', 'distract', 'sleep', 'scn', 'slc']

    for i in range(len(e1s)):
        combineBars(e1s[i], e2s[i], labels[i])