import numpy as np
import path
import hdf5storage as hdf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

'''
Another scrappy bit of code ever written
Addresses all plots created for:
Strip wise modeling of radius vs performance
Should become more gaussian and less gamma
Maybe sufficient to just show perf distributions and not the modeling?
'''

data_path = path.Path() / '..' / 'data'
tmp = hdf.loadmat(data_path / f'data_avg_radius_pre_{120}_post_{30}')

rads = tmp['radius']
perfs = tmp['performance']

perfs_20 = perfs[rads < 0.2]
perfs_40 = perfs[(rads < 0.4) & (rads >= 0.2)]
perfs_60 = perfs[(rads < 0.6) & (rads >= 0.4)]
perfs_80 = perfs[(rads < 0.8) & (rads >= 0.6)]
perfs_100 = perfs[rads >= 0.8]

def getRad(perfs):

    v, b = np.histogram(perfs, range=(-1,1), bins=21)
    b = (b[1:] + b[:-1])*0.5

    v = v / v.sum()

    return b, np.flip(v)

def plotVals(bins, v_20, v_40=None, v_60=None, v_80=None, v_100=None):

    lim = 1.0

    f, ax = plt.subplots()

    ax.plot(bins, v_20, marker='o', color='black')
    ax.axhline(0*lim+0.7, color='black', linestyle='--', alpha=0.5)
    ax.plot(bins, 1*lim+v_40, marker='o', color='black')
    ax.axhline(1*lim, color='black', linestyle='-')
    ax.axhline(1*lim+0.7, color='black', linestyle='--', alpha=0.5)
    ax.plot(bins, 2*lim+v_60, marker='o', color='black')
    ax.axhline(2*lim, color='black', linestyle='-')
    ax.axhline(2*lim+0.7, color='black', linestyle='--', alpha=0.5)
    ax.plot(bins, 3*lim+v_80, marker='o', color='black')
    ax.axhline(3*lim, color='black', linestyle='-')
    ax.axhline(3*lim+0.7, color='black', linestyle='--', alpha=0.5)
    ax.plot(bins, 4*lim+v_100, marker='o', color='black')
    ax.axhline(4*lim, color='black', linestyle='-')
    ax.axhline(4*lim+0.7, color='black', linestyle='--', alpha=0.5)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    # ax.set_yscale('log')
    ax.set_xlabel('Performance')
    ax.set_ylabel('Proportion')
    ax.set_yticks([0,0.7,1,1.7,2,2.7,3,3.7,4,4.7], labels=['0', '0.7', '0', '0.7', '0', '0.7', '0', '0.7', '0', '0.7'])

    ax.grid(False)
    ax.set_ylim(0,5.0*lim)

    sns.despine(top=True, left=True, right=False, ax=ax)

    f.savefig(f'../results/rad_stacked.pdf')
    f.savefig(f'../results/rad_stacked.png')

    plt.close(f)

    return

b, v_20 = getRad(perfs_20)
b, v_40 = getRad(perfs_40)
b, v_60 = getRad(perfs_60)
b, v_80 = getRad(perfs_80)
b, v_100 = getRad(perfs_100)

plotVals(b, v_20, v_40, v_60, v_80, v_100)
