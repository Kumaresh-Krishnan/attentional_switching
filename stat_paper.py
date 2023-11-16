import hdf5storage as hdf
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu as mn

ftype = 'data_params.mat'

def swap():

    tmp = hdf.loadmat(f'../multitrial_bl_bl/{ftype}')
    bb = tmp['all_params']

    tmp = hdf.loadmat(f'../multitrial_bl_sf/{ftype}')
    bs = tmp['all_params']

    tmp = hdf.loadmat(f'../multitrial_sf_sf/{ftype}')
    ss = tmp['all_params']

    tmp = hdf.loadmat(f'../multitrial_sf_bl/{ftype}')
    sb = tmp['all_params']

    # print(np.nanmean(sb[:,0]) / np.nanmean(ss[:,0]))
    # print(np.nanmean(ss[:,1]) / np.nanmean(sb[:,1]))

    # print(np.nanmean(bb[:,0]) / np.nanmean(bs[:,0]))
    # print(np.nanmean(bs[:,1]) / np.nanmean(bb[:,1]))

    bb_w = bb[:,1]; bs_w = bs[:,1]; sb_w = sb[:,1]; ss_w = ss[:,1]
    bb_k = 1/bb[:,0]; bs_k = 1/bs[:,0]; sb_k = 1/sb[:,0]; ss_k = 1/ss[:,0]

    # bb to bs
    print('bb to bs')
    stats_w = mn(bb_w-0.2*np.nanmean(bb_w), bs_w ,alternative='greater')
    stats_k = mn(bb_k-0.2*np.nanmean(bb_k), bs_k ,alternative='greater')
    print('w: ', stats_w, '   k: ', stats_k)

    # bb to sb
    print('bb to sb')
    stats_w = mn(bb_w-0.2*np.nanmean(bb_w), sb_w ,alternative='greater')
    stats_k = mn(bb_k-0.2*np.nanmean(bb_k), sb_k ,alternative='greater')
    print('w: ', stats_w, '   k: ', stats_k)

    # bb to ss
    print('bb to ss')
    stats_w = mn(bb_w-0.2*np.nanmean(bb_w), ss_w ,alternative='greater')
    stats_k = mn(bb_k-0.2*np.nanmean(bb_k), ss_k ,alternative='greater')
    print('w: ', stats_w, '   k: ', stats_k)

    # sb to ss
    print('sb to ss')
    stats_w = mn(sb_w-0.2*np.nanmean(sb_w), ss_w ,alternative='greater')
    stats_k = mn(sb_k-0.2*np.nanmean(sb_k), ss_k ,alternative='greater')
    print('w: ', stats_w, '   k: ', stats_k)

    # bs to ss
    print('bs to ss')
    stats_w = mn(bs_w-0.2*np.nanmean(bs_w), ss_w ,alternative='greater')
    stats_k = mn(bs_k-0.2*np.nanmean(bs_k), ss_k ,alternative='greater')
    print('w: ', stats_w, '   k: ', stats_k)

    # bs to sb
    print('bs to sb')
    stats_w = mn(bs_w-0.2*np.nanmean(bs_w), sb_w ,alternative='greater')
    stats_k = mn(bs_k-0.2*np.nanmean(bs_k), sb_k ,alternative='greater')
    print('w: ', stats_w, '   k: ', stats_k)

    return

def perturb():

    tmp = hdf.loadmat(f'../multitrial_slc_all_mut/{ftype}')
    ctrl = tmp['all_params']

    tmp = hdf.loadmat(f'../multitrial_slc_all_wt/{ftype}')
    expt = tmp['all_params']

    ctrl_w = ctrl[:,1]; expt_w = expt[:,1]
    ctrl_k = 1/ctrl[:,0]; expt_k = 1/expt[:,0]

    stats_w = mn(ctrl_w-0.2*np.nanmean(ctrl_w), expt_w ,alternative='greater')
    stats_k = mn(ctrl_k-0.2*np.nanmean(ctrl_k), expt_k ,alternative='greater')
    print('w: ', stats_w, '   k: ', stats_k)

    return

# swap()
perturb()