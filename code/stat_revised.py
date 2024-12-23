import hdf5storage as hdf
import numpy as np
import scipy.stats
import path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

sig = 0.18
theta = 15
def func(x, k1, w):

    x_val = x
    g = np.exp(-0.5*((x_val-0.5)/sig)**2)
    g = g/g.sum()

    e = (1-(x_val))**(k1-1) * np.exp(-theta*(1-(x_val))) # Gamma
    e = e / e.sum()
    
    return (1-w)*g + w*(e)

def statisticsEval(ctrl_rate, ctrl_param, expt_rate, expt_param):

    tmp = hdf.loadmat(f'../data/{ctrl_rate}')
    ctrl_perf = tmp['performance']
    tmp = hdf.loadmat(f'../data/{ctrl_param}')
    ctrl_params = tmp['params']

    tmp = hdf.loadmat(f'../data/{expt_rate}')
    expt_perf = tmp['performance']
    tmp = hdf.loadmat(f'../data/{expt_param}')
    expt_params = tmp['params']

    joint = np.concatenate((ctrl_perf, expt_perf), axis=0)
    joint = joint.reshape(joint.shape[0], -1)
    fish_idx = np.arange(joint.shape[0])
    half = int(fish_idx.shape[0]//2)

    iters = 10000

    fit_k_ctrl = np.zeros(iters)
    fit_w_ctrl = np.zeros(iters)
    fit_k_expt = np.zeros(iters)
    fit_w_expt = np.zeros(iters)

    for i in range(iters):

        np.random.shuffle(fish_idx)
        group_1 = (joint[fish_idx[:half]] + 1) / 2
        group_2 = (joint[fish_idx[half:]] + 1) / 2
        
        group_1 = group_1[~np.isnan(group_1)].ravel()
        group_2 = group_2[~np.isnan(group_2)].ravel()

        vals, b = np.histogram(group_1, bins=21, range=(0,1))
        b = (b[1:] + b[:-1])*0.5
        vals = vals / vals.sum()

        params, _ = curve_fit(func, b, vals, p0=[max(np.random.random()*6, 1e-3), \
                                                    np.random.random()], \
                                                    bounds=([1e-3, 0], [6,1]))
        
        fit_k_ctrl[i] = params[0]
        fit_w_ctrl[i] = params[1]

        vals, b = np.histogram(group_2, bins=21, range=(0,1))
        b = (b[1:] + b[:-1])*0.5
        vals = vals / vals.sum()

        params, _ = curve_fit(func, b, vals, p0=[max(np.random.random()*6, 1e-3), \
                                                    np.random.random()], \
                                                    bounds=([1e-3, 0], [6,1]))
        
        fit_k_expt[i] = params[0]
        fit_w_expt[i] = params[1]

    vals, b = np.histogram((ctrl_perf+1)/2, bins=21, range=(0,1))
    b = (b[1:] + b[:-1])*0.5
    vals = vals / vals.sum()

    ctrl_params, _ = curve_fit(func, b, vals, p0=[max(np.random.random()*6, 1e-3), \
                                                np.random.random()], \
                                                bounds=([1e-3, 0], [6,1]))

    vals, b = np.histogram((expt_perf+1)/2, bins=21, range=(0,1))
    b = (b[1:] + b[:-1])*0.5
    vals = vals / vals.sum()

    expt_params, _ = curve_fit(func, b, vals, p0=[max(np.random.random()*6, 1e-3), \
                                                np.random.random()], \
                                                bounds=([1e-3, 0], [6,1]))

    f, ax = plt.subplots()

    ax.plot(b, func(b, *ctrl_params))
    ax.set_ylim(0,0.5)
    # ax.hist(fit_k_ctrl-fit_k_expt)
    # ax.axvline(ctrl_params[0]-expt_params[0], color='black', label='ctrl')
    # ax.legend()
    plt.show()
    plt.close(f)

    # f, ax = plt.subplots()

    # ax.hist(fit_w_ctrl - fit_w_expt)
    # ax.axvline(ctrl_params[1] - expt_params[1], color='black', label='ctrl')
    # ax.legend()
    # plt.show()
    # plt.close(f)

    my_diff_k = np.abs(ctrl_params[0] - expt_params[0])
    dist_diff_k = np.abs(fit_k_ctrl - fit_k_expt)

    p_val_k = (dist_diff_k > my_diff_k).sum() / dist_diff_k.shape[0]

    my_diff_w = np.abs(ctrl_params[1] - expt_params[1])
    dist_diff_w = np.abs(fit_w_ctrl - fit_w_expt)
    p_val_w = (dist_diff_w > my_diff_w).sum() / dist_diff_w.shape[0]

    print(f'Ctrl vs. Expt: k: {p_val_k:.3f}, w: {p_val_w:.3f}')

    return
