
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import sklearn.metrics as sm

import hdf5storage as hdf
import path

sig = 0.18
theta = 15
def func(x, k1, w):

    x_val = x
    g = np.exp(-0.5*((x_val-0.5)/sig)**2)
    g = g/g.sum()

    e = (1-(x_val))**(k1-1) * np.exp(-theta*(1-(x_val))) # Gamma
    e = e / e.sum()
    
    return (1-w)*g + w*(e)

def parts(x, k1, w):

    x_val = x
    g = np.exp(-0.5*((x_val-0.5)/sig)**2)
    g = g/g.sum()

    e = (1-(x_val))**(k1-1) * np.exp(-theta*(1-(x_val))) # Gamma
    e = e / e.sum()

    return w*e, (1-w)*g

def model(data, save_name='data_params.mat'):

    data_path = path.Path() / '..' / 'data'
    tmp = hdf.loadmat(data_path / data)
    performance = (tmp['performance'] + 1) / 2

    o_vals, b = np.histogram(performance, range=(0,1), bins=21)
    b = (b[1:] + b[:-1])*0.5
    o_vals = o_vals / o_vals.sum()

    performance = np.ravel(performance)
    size = performance.shape[0]
    reps = 5000
    all_params = np.zeros((reps,2)) # Only two params - k and w
    subset_perf = np.random.choice(performance, (reps, size))

    for i in range(reps):

        vals, b = np.histogram(subset_perf[i], range=(0,1), bins=21)
        b = (b[1:] + b[:-1])*0.5
        vals = vals / vals.sum()

        params, _ = curve_fit(func, b, vals, p0=[max(np.random.random()*6, 1e-3), \
                                                np.random.random()], \
                                                bounds=([1e-3, 0], [6,1]))
        all_params[i] = params

    params = np.nanmean(all_params, axis=0)
    errs = np.nanstd(all_params, axis=0)
    print(f'{params}')
    # print(f'{np.sqrt(20/19)*np.std(o_vals - func(b, *params)):.2f}, {np.sqrt(sm.mean_squared_error(o_vals, func(b, *params))):.2f}, {sm.mean_absolute_error(o_vals, func(b, *params)):.2f}')

    hdf.savemat(data_path / save_name, {'all_params':all_params, 'params':params, 'errs':errs, 'bins':b}, \
                format='7.3', oned_as='column', store_python_metadata=True)

    f, ax = plt.subplots()

    ax.plot(2*b-1, o_vals, linestyle='--', label='expt')
    ax.plot(2*b-1, func(b, *params), linestyle='-', label='model')
    ax.set_xlabel('Performance')
    ax.set_ylabel('Proportion')
    ax.set_title(f'k={params[0]:.2f}$\pm${errs[0]:.2f} w={params[1]:.2f}$\pm${errs[1]:.2f}')
    ax.set_ylim(0,0.6)
    ax.legend()
    sns.despine(top=True, right=True)
    
    f.savefig(data_path / 'modeled_performance.pdf')
    f.savefig(data_path / 'modeled_performance.png')

    plt.close(f)

    return

def filt(data, data_rate):

    data_path = path.Path() / '..' / 'data'
    tmp = hdf.loadmat(data_path / data)
    params = tmp['params']
    b = tmp['bins']

    epart, gpart = parts(b, params[0], params[1])

    tmp = hdf.loadmat(data_path / data_rate)
    performance = (tmp['performance'] + 1) / 2
    vals, _ = np.histogram(performance, range=(0,1), bins=21)
    vals = vals / vals.sum()
    
    f, ax = plt.subplots()
    
    ax.plot(2*b-1, epart, label='attentive', color=(0.206,0.371,0.553), linestyle='--', alpha=0.7)
    ax.plot(2*b-1, gpart, label='inattentive', color=(1,0,0), linestyle='--', alpha=0.7) # (0.993,0.906,0.144)
    mix = params[1]*np.array([0.206,0.371,0.553]) + (1-params[1])*np.array([1,0,0]) # 218/256, 165/256,32/256
    ax.plot(2*b-1, func(b, *params), color=tuple(mix), label='model', alpha=0.7, marker='^', markersize=3)
    ax.plot(2*b-1, vals, label='expt', color=coloring, marker='o', markersize=3)

    ax.set_ylim(0,0.6)
    ax.set_xlim(-1.1,2.1)
    ax.set_xticks([-1,-0.5,0,0.5,1])
    ax.set_xlabel('Performance Score')
    ax.set_ylabel('Proportion')

    f.savefig(data_path / '../results' / 'parts_modeled_performance.pdf')
    f.savefig(data_path / '../results' / 'parts_modeled_performance.png')
    
    plt.close(f)

    return

def bars(data):

    data_path = path.Path() / '..' / 'data'
    tmp = hdf.loadmat(data_path / data)
    all_params = tmp['all_params']
    b = tmp['bins']
    
    # sig, lam, w

    means = np.nanmean(all_params, axis=0)
    errs = np.nanstd(all_params, axis=0)

    f, ax = plt.subplots(figsize=(2,2))
    
    ax.bar(0, 1-means[1], width=0.5, capsize=1.0, color=(0.993,0.906,0.144), linewidth=0, bottom=means[1]) # #1f77b4
    ax.bar(0, means[1], width=0.5, yerr=errs[1], capsize=1.0, color=(0.206,0.371,0.553), alpha=0.7, linewidth=0) # #1f77b4
    ax.set_ylim([0,1])
    
    ax2 = ax.twinx()
    ax2.bar(1, 1/means[0], width=0.5, yerr=errs[0], capsize=1.0, color=(0.206,0.371,0.553), alpha=0.4)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['w', f'$\kappa$'])
    ax2.set_ylim([0,2])
    ax.spines['right'].set_visible(True)

    f.savefig(data_path / '../results' / 'bar_param.pdf')
    f.savefig(data_path / '../results' / 'bar_param.png')

    plt.close(f)

    return

modeling = False
coloring = 'red' if modeling else 'black'

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

    data = 'data_params.mat'
    data_rate = 'data_bout_rate_pre_120_post_30.mat'
    # model(data_rate)
    filt(data, data_rate)
    # bars(data)
