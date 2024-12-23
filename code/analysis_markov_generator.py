import numpy as np
import path
import hdf5storage as hdf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

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

'''
Scrappiest bit of code ever written
Addresses all plots created for:
Purely state switching based generative model
This just samples from the attentive/inattentive turn distribution and generates plots
Code is copy pasted from the modeling/plotting code in the original pipeline with some edits
Ugly code, very ugly but too much efort to structure the code for this small thing!!!!!
'''

probs_engaged = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.55869824e-05,
 2.53394893e-05, 0.00000000e+00, 2.93841878e-05, 8.34592197e-05,
 1.11837079e-04, 1.89946473e-04, 1.40635100e-04, 4.33013798e-04,
 7.16213323e-04, 1.10905483e-03, 1.31821486e-03, 1.47001938e-03,
 1.43767844e-03, 1.81572433e-03, 1.64227456e-03, 2.26504997e-03,
 2.42010528e-03, 3.26889885e-03, 5.63191902e-03, 1.39522220e-02,
 8.71205662e-02, 9.53589389e-02, 5.68620418e-02, 5.13848048e-02,
 6.83414741e-02, 8.69303001e-02, 1.02479535e-01, 1.09021764e-01,
 1.04391043e-01, 7.87993624e-02, 5.12708927e-02, 3.09406810e-02,
 1.68971952e-02, 9.09934351e-03, 5.28061728e-03, 2.36376580e-03,
 2.22617564e-03, 2.32623582e-03, 5.09309517e-04, 1.41348934e-04,
 2.83427130e-05, 2.83637465e-05, 5.76636545e-05, 2.52924423e-05,
 2.83637465e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]

probs_disengaged = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 3.44367902e-05, 5.47793912e-05, 0.00000000e+00, 5.14027161e-05,
 0.00000000e+00, 0.00000000e+00, 8.23271415e-05, 3.39827961e-05,
 7.41468960e-05, 1.34640844e-04, 4.08249351e-04, 5.63119214e-04,
 1.28255742e-03, 2.34343109e-03, 2.73676314e-03, 3.64816727e-03,
 8.26440978e-03, 1.24549014e-02, 2.09919970e-02, 2.85465239e-02,
 3.52141833e-02, 4.64853179e-02, 4.87837790e-02, 4.49938384e-02,
 4.56954785e-02, 3.92660269e-02, 5.43219434e-02, 1.05463374e-01,
 1.37264189e-01, 6.50060661e-02, 3.78135137e-02, 2.87714548e-02,
 2.86305895e-02, 3.30646215e-02, 3.24027158e-02, 3.13940496e-02,
 2.85183972e-02, 2.41924790e-02, 1.86686423e-02, 1.31163994e-02,
 7.81364124e-03, 3.51627961e-03, 3.20009233e-03, 2.04749719e-03,
 1.30998211e-03, 3.81252230e-04, 1.61104113e-04, 1.41282594e-04,
 4.15070555e-04, 3.79960911e-05, 0.00000000e+00, 1.15706155e-04,
 0.00000000e+00, 3.63356679e-05, 0.00000000e+00, 5.08657174e-05,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]

angles = [-180., -174.92957746, -169.85915493, -164.78873239, -159.71830986,
 -154.64788732, -149.57746479, -144.50704225, -139.43661972, -134.36619718,
 -129.29577465, -124.22535211, -119.15492958, -114.08450704, -109.01408451,
 -103.94366197,  -98.87323944,  -93.8028169,   -88.73239437,  -83.66197183,
  -78.5915493,   -73.52112676,  -68.45070423,  -63.38028169,  -58.30985915,
  -53.23943662,  -48.16901408,  -43.09859155, -38.02816901,  -32.95774648,
  -27.88732394,  -22.81690141,  -17.74647887,  -12.67605634,   -7.6056338,
   -2.53521127,    2.53521127,    7.6056338,   12.67605634,   17.74647887,
   22.81690141,   27.88732394,   32.95774648,   38.02816901,   43.09859155,
   48.16901408,   53.23943662,   58.30985915,   63.38028169,   68.45070423,
   73.52112676,   78.5915493,    83.66197183,   88.73239437,   93.8028169,
   98.87323944,  103.94366197,  109.01408451,  114.08450704,  119.15492958,
  124.22535211,  129.29577465,  134.36619718,  139.43661972,  144.50704225,
  149.57746479,  154.64788732,  159.71830986,  164.78873239,  169.85915493,
  174.92957746,  180.        ]

fish = 64
trials = 30

tmp = hdf.loadmat(path.Path() / '..' / 'data' / 'data_bout_rate_pre_120_post_30.mat')
r1s = (tmp['bout_rate'] / 60)*1.0
r2s = (tmp['stim_rate'] / 60)*1.2
r1s = np.reshape(r1s, (64,-1), order='F')
r2s = np.reshape(r2s, (64,-1), order='F')
r1s[np.isnan(r1s)] = 1.
r2s[np.isnan(r2s)] = 1.2

perfs = np.zeros(r1s.shape)

for f in range(fish):
    
    state = np.random.choice([0,1])

    for t in range(trials):
        toss = np.random.random()
        bouts = []

        if state == 1 and toss < 0.2:
            state = 1 - state
        elif state == 0 and toss >= 0.2:
            state = 1 - state
        
        for b in range(int(r2s[f,t]*30)):

            if state == 0:
                bout = np.random.choice(angles, p=probs_disengaged)
            else:
                bout = np.random.choice(angles, p=probs_engaged)

            angs = [bout, bout+1, bout+2, bout+3, bout+4, bout+5]
            bouts.append(np.random.choice(angs))

        bouts = np.array(bouts)

        if state == 0:
            target = np.random.choice([0,1])

            if target == 0:
                score = 2*(((bouts > 0).sum() + 0.5*(bouts == 0).sum()) / bouts.shape[0]) - 1
            else:
                score = 2*(((bouts < 0).sum() + 0.5*(bouts == 0).sum()) / bouts.shape[0]) - 1

        else:
            score = 2*(((bouts > 0).sum() + 0.5*(bouts == 0).sum()) / bouts.shape[0]) - 1

        perfs[f,t] = score

f, ax = plt.subplots()
        
ax.scatter(r1s[:,:].ravel()*60, perfs[:,:].ravel(), alpha=0.05, color='red')

ax.set_xlabel('Baseline bout rate (bouts/min)')
ax.set_ylabel('Performance')

ax.grid(False)
ax.set_xlim(0,150)
ax.set_ylim(-1.1,1.1)
sns.despine(top=True, right=True)

f.savefig('../results/state_scatter.pdf')
f.savefig('../results/state_scatter.png')
plt.close(f)


v, b = np.histogram(perfs, range=(-1,1), bins=21)
b = (b[1:] + b[:-1])*0.5

v = v / v.sum()

f, ax = plt.subplots()

ax.plot(b, v, marker='o', color='red')

ax.set_xlabel('Performance')
ax.set_ylabel('Count')
#ax.set_title('Distribution of performance scores')
ax.grid(False)
ax.set_ylim(0,0.6)

sns.despine(top=True, right=True, ax=ax)

f.savefig('../results/state_perf_distribution.pdf')
f.savefig('../results/state_perf_distribution.png')

plt.close(f)


def model(perfs):

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

        params, _ = curve_fit(func, b, vals, p0=[max(np.random.random()*6, 1e-3), \
                                                np.random.random()], \
                                                bounds=([1e-3, 0], [6,1]))
        all_params[i] = params

    params = np.nanmean(all_params, axis=0)
    errs = np.nanstd(all_params, axis=0)

    data_params = {'all_params':all_params, 'params':params, 'errs':errs, 'bins':b}

    filt(data_params, perfs)
    bars(data_params)

sig = 0.07
theta = 17
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

def filt(tmp, performance):

    params = tmp['params']
    b = tmp['bins']

    epart, gpart = parts(b, params[0], params[1])

    performance = (performance + 1) / 2
    vals, _ = np.histogram(performance, range=(0,1), bins=21)
    vals = vals / vals.sum()
    
    f, ax = plt.subplots()
    
    ax.plot(2*b-1, epart, label='attentive', color=(0.206,0.371,0.553), linestyle='--', alpha=0.7)
    ax.plot(2*b-1, gpart, label='inattentive', color='goldenrod', linestyle='--', alpha=0.7) # (0.993,0.906,0.144)
    mix = params[1]*np.array([0.206,0.371,0.553]) + (1-params[1])*np.array([218/256, 165/256,32/256])
    ax.plot(2*b-1, func(b, *params), color=tuple(mix), label='model', alpha=0.7, marker='^', markersize=3)
    ax.plot(2*b-1, vals, label='expt', color='red', marker='o', markersize=3)

    ax.set_ylim(0,0.6)
    ax.set_xlim(-1.1,2.1)
    ax.set_xticks([-1,-0.5,0,0.5,1])
    ax.set_xlabel('Performance Score')
    ax.set_ylabel('Proportion')

    f.savefig('../results/state_parts_modeled_performance.pdf')
    f.savefig('../results/state_parts_modeled_performance.png')
    
    plt.close(f)

    return

def bars(tmp):

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

    f.savefig('../results/state_bar_param.pdf')
    f.savefig('../results/state_bar_param.png')

    plt.close(f)

    return

model(perfs)