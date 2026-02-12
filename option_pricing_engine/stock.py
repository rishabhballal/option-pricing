import math
import numpy as np
from . import rng

dt = 1/252

# for binomial trees
nr_steps = 252*4

# for Monte Carlo simulations
nr_paths = 10**5

class GeometricBrownianMotion:
    def __init__(self, spot, rate, divid, vol):
        self.spot = spot
        self.rate = rate
        self.divid = divid
        self.vol = vol

    def tree(self, time=252):
        self.up_ = math.exp(self.vol * math.sqrt(time * dt / nr_steps))
        self.down_ = 1 / self.up_
        self.pr_ = (math.exp((self.rate - self.divid) * time * dt / \
            nr_steps) - self.down_) / (self.up_ - self.down_)
        tree = [[self.spot] * (i + 1) for i in range(nr_steps + 1)]
        for i in range(1, nr_steps + 1):
            for j in range(i + 1):
                tree[i][j] *= self.up_**(i - j) * self.down_**j
        return tree

    def paths(self, time=252, rand=[]):
        if not len(rand):
            rand = rng.standard_normal((time, nr_paths))
        paths = np.ones((1 + time, nr_paths)) * self.spot
        for i in range(time):
            paths[i+1] = paths[i] * np.exp((self.rate - self.divid) * dt - \
                self.vol**2 * dt / 2 + self.vol * math.sqrt(dt) * rand[i])
        return paths
