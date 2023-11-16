#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  sampler.py
#  
#  Copyright 2019 Kumaresh <kumaresh_krishnan@g.harvard.edu>
#  
#  version 1.0

import numpy as np
import os, sys

from numba import njit, prange, int32, float32, types, typed
from numba.experimental import jitclass

import matplotlib.pyplot as plt
import seaborn as sns

key_val = (types.unicode_type, float32[:])

params = [('bound', float32), ('r1', float32), \
    ('r2', float32), ('size', int32), \
    ('x', float32[:]), ('bouts', float32[:]), \
    ('previous', int32), ('dt', float32), \
    ('dist', types.DictType(*key_val))]

@jitclass(params)
class Sampler:

    def __init__(self, x, d1, d2, d3, d4, d5, b=1.0, r1=0.7, r2=2.0):

        self.bound = b
        self.r1 = r1
        self.r2 = r2
        self.size = x.shape[0]
        self.previous = -30
        self.dt = 0.01
        
        self.bouts = np.zeros(self.size, dtype=float32)
        self.dist = typed.Dict.empty(*key_val)
        self.dist['base'] = d1
        self.dist['left'] = d2
        self.dist['right'] = d3
        self.dist['base_left'] = d4
        self.dist['base_right'] = d5

        self.x = x.copy()

    def nbound(self, b):

        self.bound = b
    
    def nr1(self, r):

        self.r1 = r

    def nr2(self, r):

        self.r2 = r

    def run(self):

        for t in range(self.size):

            if self.x[t] > self.bound:
                r = self.r2 * self.dt
                mode = 'right'
            elif self.x[t] < -self.bound:
                r = self.r2 * self.dt
                mode = 'left'
            else:
                r = self.r1 * self.dt
                if self.x[t] >= 0:
                    mode = 'base_right'
                    alt_mode = 'base_left'
                else:
                    mode = 'base_left'
                    alt_mode = 'base_right'
            
            if np.random.random() < r:
                if np.random.random() < 0.98:
                    angle = np.random.choice(self.dist[mode])
                else:
                    angle = np.random.choice(self.dist[alt_mode])

                self.bouts[t] = np.sign(angle) + 0.5*(angle == 0)

            if self.bouts[t] != 0:
                if t - self.previous < 21: # 200 ms between bouts
                    self.bouts[t] = 0
                else:
                    self.previous = t

if __name__ == '__main__':

    pass
