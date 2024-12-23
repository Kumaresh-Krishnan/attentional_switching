import path
import matplotlib.pyplot as plt
import seaborn as sns
import hdf5storage as hdf
import numpy as np
from scipy.optimize import curve_fit

def binomial():

    data_path = path.Path() / '..' / 'data'

    tmp = hdf.loadmat(data_path / f'baseline_data_bout_rate_pre_120_post_30')
    numbouts = tmp['incorrect_data']
    performance = tmp['performance']

    filt = ~np.isnan(numbouts) & (numbouts > 0)

    possible_vals = numbouts[filt].ravel().astype(int)

    #ps = np.linspace(0,0.9,50)
    ps = [0,0.25,0.5,0.75,1.0]

    reps = 50
    bins= 21

    perfs = np.zeros((len(ps), reps, bins))

    f, ax = plt.subplots()

    for idx_p, p in enumerate(ps):
        # print(p) # Prob to bout in same direction

        for rep in range(reps):
            targets = np.random.choice([-1,1], size=possible_vals.shape[0])

            scores = np.zeros(possible_vals.shape[0])

            for idx, val in enumerate(possible_vals):

                rands = np.random.random(size=int(val))
                bouts = np.zeros(val)
                
                bouts[0] = np.random.choice([-1,1])

                for bout_num in range(1, val):
                    bouts[bout_num] = bouts[bout_num-1] if rands[bout_num] <= p else -1*bouts[bout_num-1]

                scores[idx] = ((bouts == targets[idx]).sum() / val)*2 - 1

            v, b = np.histogram(scores, range=(-1,1), bins=bins)
            v = v/v.sum()
            perfs[idx_p, rep] = v

    avg_perfs = perfs.mean(axis=1)
    b = np.histogram_bin_edges([0], range=(-1,1), bins=bins)
    b = 0.5*(b[1:] + b[:-1])

    # for r in range(reps):
    #     ax.plot(b, perfs[3,r,:], label=f'0.75', marker='o', markersize=2, alpha=0.05, color='red')
    
    for idx, v in enumerate(avg_perfs):
        ax.plot(b, v, label=f'{ps[idx]}', marker='o', markersize=2, alpha=max(1-ps[idx], 0.05), color='red')

    ax.set_xlabel('Performance')
    ax.set_ylabel('Probability')
    #ax.set_title('Averaged Probability Distribution By NumBouts')
    ax.grid(False)

    v, b = np.histogram(performance, range=(-1,1), bins=bins)
    b = 0.5*(b[1:] + b[:-1])
    v = v / v.sum()
    ax.plot(b, v, marker='o', markersize=2, label='Expt', color='black')
    ax.legend()
    sns.despine(top=True, right=True)

    f.savefig(f'../results/binomial_compare.pdf')
    f.savefig(f'../results/binomial_compare.png')
    # plt.show()
    plt.close(f)

    #hdf.savemat('perf_data_full.mat', {'perf':avg_perfs}, format='7.3', oned_as='column', store_python_metadata=True)

    return

def func(x, sig):

    val = np.exp(-0.5*((x-0.5)/sig)**2)
    val = val / val.sum()

    return val

def fit():

    tmp = hdf.loadmat('perf_data.mat')
    perf = tmp['perf']

    b = np.histogram_bin_edges([0], range=(0,1), bins=20)
    b = 0.5*(b[1:] + b[:-1])
    probs = [0, 0.25, 0.5, 0.75, 1.0]

    for idx, p in enumerate(perf):
        f, ax = plt.subplots()
        params, _ = curve_fit(func, b, p, p0=[max(np.random.random(), 1e-3)], bounds=([1e-3], [0.5]))

        ax.plot(b, func(b, *params))
        ax.plot(b, p)
        ax.set_ylim(0,0.7)
        ax.set_title(f'$\sigma$={params[0]}')
        sns.despine(top=True, right=True)

        f.savefig(f'fit_p_{probs[idx]}.png')
        f.savefig(f'fit_p_{probs[idx]}.pdf')
        plt.close(f)

    return

def relation():

    tmp = hdf.loadmat('perf_data_full.mat')
    perf = tmp['perf']

    b = np.histogram_bin_edges([0], range=(0,1), bins=20)
    b = 0.5*(b[1:] + b[:-1])
    
    probs = np.linspace(0,0.9,50)
    sigs = np.zeros(len(probs))

    for idx, p in enumerate(perf):
        
        params, _ = curve_fit(func, b, p, p0=[max(np.random.random(), 1e-3)], bounds=([1e-3], [1.0]))
        sigs[idx] = params[0]

    f, ax = plt.subplots()

    ax.plot(probs, sigs); ax.axhline(0.2, color='black', linestyle='--'); ax.axvline(0.75, color='black', linestyle='--')
    ax.set_xlabel(f'p')
    ax.set_ylabel(f'$\sigma$')
    sns.despine(top=True, right=True)
    
    f.savefig('p_sigma.png')
    f.savefig('p_sigma.pdf')
    plt.close(f)

    hdf.savemat('sigma_params', {'params': sigs}, format='7.3', oned_as='column', store_python_metadata=True)

    return

def findStreaks(vals, bins, p):

    l_idx = (vals < 0).astype(int)

    streaks = []
    #probs = []
    ctr = 1
    for i in range(1,l_idx.shape[0]):
        if l_idx[i] - l_idx[i-1] != 0:
            streaks.append(ctr)
            #probs.append(binom.pmf(ctr, vals.shape[0], 0.5))
            ctr = 1
        else:
            ctr += 1
    streaks.append(ctr)
    #probs.append(binom.pmf(ctr, vals.shape[0], 0.5))

    # s, _ = np.histogram(streaks, range=(0,bins+1), bins=bins)
    # s = s/s.sum()

    return streaks#, probs

def computeStreaks():

    data_path = path.Path() / '..' / '..' / 'decision_paper' / 'data' / 'multitrial_50_ctrl'

    tmp = hdf.loadmat(data_path / f'data_bout_rate_pre_120_post_30')
    numbouts = tmp['incorrect_data']

    tmp = hdf.loadmat(data_path / f'data_streaks_pre_120_post_30')
    expt = tmp['streaks']
    
    filt = ~np.isnan(numbouts) & (numbouts > 0)

    possible_vals = numbouts[filt].ravel().astype(int)

    ps = [0,0.25,0.5,0.75,1.0]
    #ps = np.linspace(0,0.9,50)

    reps = 1
    bins= 25

    streaks = np.zeros((len(ps), reps, bins))

    f, ax = plt.subplots()

    for idx_p, p in enumerate(ps):
        print(p) # Prob to bout in same direction

        for rep in range(reps):

            scores = []#np.zeros((possible_vals.shape[0], bins))
            #probs = []

            for idx, val in enumerate(possible_vals):
                rands = np.random.random(size=int(val))
                bouts = np.zeros(val)
                
                bouts[0] = np.random.choice([-1,1])

                for bout_num in range(1, val):
                    bouts[bout_num] = bouts[bout_num-1] if rands[bout_num] <= p else -1*bouts[bout_num-1]

                #scores[idx] = findStreaks(bouts, bins)
                sc= findStreaks(bouts, bins, p)
                scores.extend(sc)
                #probs.extend(pr)
            
            #v, b, _ = binned_statistic(scores, probs, statistic=np.nanmean, range=(0,bins+1), bins=bins)
            v, b = np.histogram(scores, bins=bins, range=(0,bins+1))
            v = v/v.sum()
        
            streaks[idx_p, rep] = v#np.nanmean(scores, axis=0)

    
    avg_streaks = np.nanmean(streaks, axis=1)
    v, b = np.histogram(expt, range=(0,bins+1), bins=bins)
    v = v/v.sum()
    b = 0.5*(b[1:] + b[:-1])

    for idx, row in enumerate(avg_streaks):
        ax.plot(b, row, label=f'{ps[idx]}', color='red', alpha=max((1-ps[idx]), 0.05))
    ax.plot(b, v, linestyle='None', marker='o', markersize=3, color='black')

    ax.legend()
    ax.set_xlabel('Streak length')
    ax.set_ylabel('Proportion')
    #ax.set_ylim(0.3,1.0)
    sns.despine(top=True, right=True)

    # f.savefig('streak_dist_constant.png')
    # f.savefig('streak_dist_constant.pdf')
    plt.show()
    plt.close(f)
    
    #hdf.savemat('streak_data_full.mat', {'streaks': streaks, 'avg': avg_streaks, 'bins': b}, format='7.3', oned_as='column', store_python_metadata=True)

    return

def eFunc(x, lam):

    vals = np.exp(-lam*x)
    vals = vals/vals.sum()

    return vals

def streakRelation():

    tmp = hdf.loadmat('streak_data_full.mat')
    streaks = tmp['avg']
    b = tmp['bins']

    lams = np.zeros(streaks.shape[0])
    ps = np.linspace(0,0.9,50)

    for idx, row in enumerate(streaks):
        params, _ = curve_fit(eFunc, b, row, p0=[max(np.random.random(), 1e-3)], bounds=([1e-3], [10]))
        lams[idx] = params[0]

    f, ax = plt.subplots()

    ax.plot(ps, lams)
    ax.set_xlabel('p')
    ax.set_ylabel(f'$\lambda$')
    sns.despine(top=True, right=True)

    f.savefig('p_lambda.png')
    f.savefig('p_lambda.pdf')
    plt.close(f)

    hdf.savemat('lambda_params', {'params': lams}, format='7.3', oned_as='column', store_python_metadata=True)

    return


def overlay():

    tmp = hdf.loadmat('sigma_params.mat')
    sigs = tmp['params']

    tmp = hdf.loadmat('lambda_params.mat')
    lams = tmp['params']

    ps = np.linspace(0,0.9,50)

    f,ax = plt.subplots()
    
    ax2 = ax.twinx()

    ax.plot(ps, sigs, color='red', label=f'$\sigma$')
    ax2.plot(ps, np.flip(lams), color='black', label=f'$\lambda$')

    ax.set_xlabel('p')
    ax.set_ylabel('$\sigma$')
    ax2.set_ylabel('$\lambda$')
    sns.despine(top=True, right=False)
    ax.legend(loc=4)
    ax2.legend(loc=0)

    f.savefig('match_params.png')
    f.savefig('match_params.pdf')
    plt.close(f)


    return

binomial()
# fit()
# relation()
# computeStreaks()
# streakRelation()
# overlay()

# f, ax = plt.subplots()

# x_range = np.arange(25)

# ax.plot(x_range, 0.5**x_range)
# plt.show()

