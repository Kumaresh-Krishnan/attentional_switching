"""
"""
# 0.0 Imports
from pathlib import Path
import hdf5storage as hdf
import autograd.numpy as np
import autograd.numpy.random as npr
# set seed for reproducibility
npr.seed(0)
import matplotlib.pyplot as plt
import pandas as pd
import ssm

# 1.0 Import data
# Mat files created using hdf.savemat(<path>, {dict}, format='7.3', oned_as=column, store_python_metadata=True)
# don't use pathlib Paths as hdf5storage doesn't like them
performance = hdf.loadmat('./data/data_performance.mat')['performance']
rates = hdf.loadmat('./data/data_bout_rates.mat')['rates']

# 1.1 Preprocess data for ssm
# remove all rows that have a nan
performance = performance[~np.isnan(performance).any(axis=1)]
rates = rates[~np.isnan(rates).any(axis=1)]

# normalize 
performance_norm = (performance - np.mean(performance, axis=0)) / np.std(performance, axis=0)
rates_norm = (rates - np.mean(rates, axis=0)) / np.std(rates, axis=0)

# format data for ssm observations
obs = []
for i in range(performance_norm.shape[0]):
    obs.append(np.expand_dims(performance_norm[i], axis=-1))

# 2.0 Fit model with performance observations
D = 1 # number of observed dimensions
n_cv = 5 # number of cross validation folds
cval_edges = np.linspace(0, performance.shape[0], n_cv+1).astype('int')
results = [] # list to store results
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for K in range(1, 10):
    n_params = D*K + D*(D+1)/2*K + K*(K-1)  # means, covs, transition matrices
    
    for i in range(n_cv):
        # train test split
        test_inds = np.arange(cval_edges[i], cval_edges[i+1])
        train_inds = np.delete(np.arange(performance.shape[0]), test_inds)
        obs_train = [obs[i] for i in train_inds]
        obs_test = [obs[i] for i in test_inds]
        # create hmm object and fit
        hmm = ssm.HMM(K, D, observations="gaussian")
        train_lls = hmm.fit(obs_train, method="em", num_iters=150)
        test_ll = hmm.log_probability(obs_test)
        bic = -2*test_ll + n_params*np.log(0.5*len(obs_train))
        ax.scatter(K, bic, c='k')

        # find most likely states
        zs = []
        for trial in obs:
            z = hmm.most_likely_states(trial)
            zs.append(z)
        zs = np.array(zs)

        # append fit to results
        results.append({
            'K': K,
            'n_cv': i,
            'train_lls': train_lls,
            'test_ll': test_ll,
            'bic': bic,
            'zs': zs,
            'hmm': hmm,
            'means': hmm.observations.mus,
            'covs': hmm.observations.Sigmas
        })

ax.set(xlabel='Number of Hidden States',
       ylabel='Bayesian Information Criterion',
       xlim=[0, 10])

# save results to file
results_dir = Path('./results')
if not results_dir.exists():
    results_dir.mkdir()
fig.savefig(Path('./results/bic.png'))
results_df = pd.DataFrame(results)
results_df.to_csv(Path('./results/all_fits_p.csv'))
results_df.to_pickle(Path('./results/all_fits_p.pkl'))

# uncomment to load pre-fit results from file
# results_df = pd.read_pickle('./results/models/all_fits_p.pkl')

# for each row in results_df plot the most likely states
K = 2
sorting_inds = np.flip(np.argsort(np.mean(performance, axis=1)))
performance_sorted = performance[sorting_inds, :]
rates_sorted = rates[sorting_inds, :]
iter = 0
for i, result in results_df.iterrows():
    if result['K'] == K:
        iter += 1
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        ax = axes[0,0]
        im_p = ax.imshow(performance_sorted, aspect='auto', cmap='bwr')
        ax.set(xlabel='Trial',
            ylabel='Fish',
            title='Performance')
        plt.colorbar(im_p, ax=ax)
        
        ax = axes[0,1]
        ax.imshow(result['zs'][sorting_inds, :], aspect='auto', cmap='Dark2', vmin=0, vmax=7, interpolation='none')
        ax.set(xlabel='Trial',
            ylabel='Fish',
            title='Most Likely States')

        ax = axes[1,0]
        im_bbr = ax.imshow(rates_sorted, aspect='auto', cmap='viridis')
        ax.set(xlabel='Trial',
            ylabel='Fish',
            title='Baseline Bout Rates')
        plt.colorbar(im_bbr, ax=ax)
        
        ax = axes[1,1]
        ax.scatter(rates.flatten(), performance.flatten(), s=12, alpha=0.3, c=result['zs'].flatten(), cmap='Dark2', vmin=0, vmax=7)
        ax.set(xlabel='Baseline Bout Rate',
            ylabel='Performance',)

        fig.suptitle(f'{K}-State Model: Engaged (orange) vs Disengaged (green)', fontsize=16)
        fig.tight_layout()

        # save fig
        fig.savefig(Path(f'./results/state_assignments_iter{iter}.png'))

# for K=2 plot the state dwell time distributions
K = 2
state_list_all = np.array([])
state_durations_all = np.array([])
for i, result in results_df.iterrows():
    if result['K'] == K:
        zs = result['zs']
        # iterate through each row in zs and append to state list and durations
        for row in zs:
            state_list, state_durations = ssm.util.rle(row)
            state_list_all = np.append(state_list_all, state_list)
            state_durations_all = np.append(state_durations_all, state_durations)
        
        # stack durations into list for plotting
        state_durs_stacked = []

        for s in range(K):
            state_durs_stacked.append(state_durations_all[state_list_all == s])

        fig, ax = plt.subplots(figsize=(8, 4))
        # plot each item in state_durs_stacked using Dark2 as colormap
        # ax.hist(state_durs_stacked, label=['state ' + str(s) for s in range(K)], cmap='Dark2', bins=20)

        colors = plt.cm.Dark2(np.linspace(0, 1, 7))
        for i, state_durs in enumerate(state_durs_stacked):
            ax.hist(state_durs, label='state ' + str(i), color=colors[i], alpha=0.7, bins=range(20), histtype='step', density=True, linewidth=2)

        ax.set(xlabel='State Duration (trials)',
               xticks=np.arange(0, 20, 2),
               ylabel='Density',
               title='State Dwell Time Distribution')
        ax.legend()

        break
# save results to file
fig.savefig(Path('./results/state_dwell_time_dist.png'))