#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  integrator.py
#  
#  Copyright 2019 Kumaresh <kumaresh_krishnan@g.harvard.edu>
#  
#  version 1.0

import numpy as np
import os, sys

from numba import njit, int32, float32, prange
from numba.experimental import jitclass

import matplotlib.pyplot as plt
import seaborn as sns

params = [('sigma', float32), ('trace', float32[:]), \
    ('c', float32[:]), ('dt', float32), \
    ('tau', float32), ('t', int32), \
    ('size', int32)]

@jitclass(params)
class Integrator:

    def __init__(self, t=180, sig=20.0, tau=2.0):
        self.sigma = sig
        self.t = t
        self.tau = tau
        self.dt = 0.01
        self.size = self.t // self.dt

        self.trace = np.zeros(self.size, dtype=float32)
        self.c = np.zeros(self.size, dtype=float32)

        
    def nSigma(self, s):

        self.sig = s

    def nTau(self, tau):

        self.tau = tau

    def nC(self, c, start, end):

        self.c[start//self.dt : end//self.dt] = c

    def nNoise(self, n):

        self.noise = n.copy()

    def run(self):

        for i in range(self.size - 1):
            dX = -self.trace[i] + self.c[i]  + np.random.normal(0, self.sigma)
            self.trace[i+1] = self.trace[i] + dX*self.dt / self.tau
            # if self.trace[i+1] > 1e4:
            #     self.trace[i+1] = 1e4
            # elif self.trace[i+1] < -1e4:
            #     self.trace[i+1] = -1e-4
            # elif np.abs(self.trace[i+1]) < 1e-4:
            #     self.trace[i+1] = 1e-4

    def results(self):

        return self.trace


if __name__ == '__main__':

    x = Integrator(t=60, tau=2.0, sig=5)
    x.nC(0.25, 30, 60)
    
    x.run()

    sns.set_style('white')
    sns.set_style('ticks')

    f, ax = plt.subplots()

    ax.plot(np.arange(x.size)/100, x.results(), label='trace')
    ax.axhline(1.0, linestyle='--', color='grey', alpha=0.7, label='thresh')
    ax.axhline(-1.0, linestyle='--', color='grey', alpha=0.7)
    ax.axhline(0, linestyle='--', color='grey', alpha=0.3, label='Zero')
    ax.axvspan(30, 60, color='grey', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X(t)')
    ax.set_ylim(-1.6,1.6)
    ax.set_title(f'Sample trajectory: $\\tau$={x.tau:.2f} $\sigma$={x.sigma:.2f}')
    ax.legend()
    ax.grid(False)
    sns.despine(top=True, right=True)

    plt.show()
    
    sys.exit()
