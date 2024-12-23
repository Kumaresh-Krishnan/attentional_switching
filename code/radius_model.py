import numpy as np
import path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import hdf5storage as hdf


def model(perfs, strip, extra=False):

    performance = (perfs + 1) / 2

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
        
        if extra:
            params, _ = curve_fit(func, b, vals, p0=[max(np.random.random()*1.5, 1e-3), \
                                                np.random.random()*0.1], \
                                                bounds=([1e-3, 0], [1.5,0.1]))
        else:

            params, _ = curve_fit(func, b, vals, p0=[max(np.random.random()*6, 1e-3), \
                                                    np.random.random()], \
                                                    bounds=([1e-3, 0], [6,1]))
        all_params[i] = params

    params = np.nanmean(all_params, axis=0)
    errs = np.nanstd(all_params, axis=0)

    data_params = {'all_params':all_params, 'params':params, 'errs':errs, 'bins':b}

    filt(data_params, perfs, strip)
    bars(data_params, strip)
    
    return data_params

sig = 0.18
theta = 15
mu = 0.5
def func(x, k1, w):

    x_val = x
    g = np.exp(-0.5*((x_val-mu)/sig)**2)
    g = g/g.sum()

    e = (1-(x_val))**(k1-1) * np.exp(-theta*(1-(x_val))) # Gamma
    e = e / e.sum()
    
    return (1-w)*g + w*(e)

def parts(x, k1, w):

    x_val = x
    g = np.exp(-0.5*((x_val-mu)/sig)**2)
    g = g/g.sum()

    e = (1-(x_val))**(k1-1) * np.exp(-theta*(1-(x_val))) # Gamma
    e = e / e.sum()

    return w*e, (1-w)*g

def filt(tmp, performance, strip):

    params = tmp['params']
    b = tmp['bins']

    epart, gpart = parts(b, params[0], params[1])

    performance = (performance + 1) / 2
    vals, _ = np.histogram(performance, range=(0,1), bins=21)
    vals = vals / vals.sum()
    
    f, ax = plt.subplots()
    
    ax.plot(2*b-1, epart, label='attentive', color=(0.206,0.371,0.553), linestyle='--', alpha=0.7)
    ax.plot(2*b-1, gpart, label='inattentive', color='red', linestyle='--', alpha=0.7) # (0.993,0.906,0.144)
    mix = params[1]*np.array([0.206,0.371,0.553]) + (1-params[1])*np.array([1,0,0])
    ax.plot(2*b-1, func(b, *params), color=tuple(mix), label='model', alpha=0.7, marker='^', markersize=3)
    ax.plot(2*b-1, vals, label='expt', color='red', marker='o', markersize=3)

    ax.set_ylim(0,0.6)
    ax.set_xlim(-1.1,2.1)
    ax.set_xticks([-1,-0.5,0,0.5,1])
    ax.set_xlabel('Performance Score')
    ax.set_ylabel('Proportion')

    f.savefig(f'../multitrial_2/rad_{strip}_parts_modeled_performance.pdf')
    f.savefig(f'../multitrial_2/rad_{strip}_parts_modeled_performance.png')
    
    plt.close(f)

    return

def bars(tmp, strip):

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
    ax2.set_ylim([0,2.5])
    ax.spines['right'].set_visible(True)

    f.savefig(f'../multitrial_2/rad_{strip}_bar_param.pdf')
    f.savefig(f'../multitrial_2/rad_{strip}_bar_param.png')

    plt.close(f)

    return

data_path = path.Path() / '..' / 'data'
tmp = hdf.loadmat(data_path / f'data_avg_radius_pre_{120}_post_{30}')

rads = tmp['radius']
perfs = tmp['performance']

perfs_20 = perfs[rads < 0.2]
perfs_40 = perfs[(rads < 0.4) & (rads >= 0.2)]
perfs_60 = perfs[(rads < 0.6) & (rads >= 0.4)]
perfs_80 = perfs[(rads < 0.8) & (rads >= 0.6)]
perfs_100 = perfs[rads >= 0.8]

params_20 = model(perfs_20, 20)
params_40 = model(perfs_40, 40)
params_60 = model(perfs_60, 60)
params_80 = model(perfs_80, 80)
params_100 = model(perfs_100, 100, True)


def combinedBars(p20, p40, p60, p80, p100):

    p_20m = np.nanmean(p20, axis=0)
    p_40m = np.nanmean(p40, axis=0)
    p_60m = np.nanmean(p60, axis=0)
    p_80m = np.nanmean(p80, axis=0)
    p_100m = np.nanmean(p100, axis=0)

    p_20e = np.nanstd(p20, axis=0)
    p_40e = np.nanstd(p40, axis=0)
    p_60e = np.nanstd(p60, axis=0)
    p_80e = np.nanstd(p80, axis=0)
    p_100e = np.nanstd(p100, axis=0)

    f, ax = plt.subplots()

    ax.bar([0,1,2,3,4], [1-p_20m[1], 1-p_40m[1], 1-p_60m[1], 1-p_80m[1], 1-p_100m[1]], width=0.5, capsize=1.0, color=(1., 0, 0),\
           linewidth=0, bottom=[p_20m[1], p_40m[1], p_60m[1], p_80m[1], p_100m[1]])
    ax.bar([0,1,2,3,4], [p_20m[1], p_40m[1], p_60m[1], p_80m[1], p_100m[1]], yerr=[p_20e[1], p_40e[1], p_60e[1], p_80e[1], p_100e[1]],\
           color=(0.206,0.371,0.553), capsize=1.0, alpha=0.7)
    
    ax.set_xticklabels(['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
    ax.set_ylim(0,1)

    f.savefig(f'../results/strip_bar_w.pdf')
    plt.close(f)

    f, ax = plt.subplots()

    ax.bar([0,1,2,3,4], [p_20m[0], p_40m[0], p_60m[0], p_80m[0], p_100m[0]], yerr=[p_20e[0], p_40e[0], p_60e[0], p_80e[0], p_100e[0]],\
           color=(0.206,0.371,0.553), capsize=1.0, alpha=0.7)
    
    ax.set_xticklabels(['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
    ax.set_ylim(0,2.5)

    f.savefig(f'../results/strip_bar_k.pdf')
    plt.close(f)

    return

combinedBars(params_20['all_params'], params_40['all_params'], params_60['all_params'], params_80['all_params'], params_100['all_params'])