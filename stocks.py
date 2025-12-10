import math
import numpy as np

dt = 1/252

# for monte carlo
rng = np.random.default_rng()
nr_paths = 10**4

# for binomial tree
nr_steps = 1000

class Stock:
    def __init__(self, spot, rate, divid, vol):
        self.spot = spot
        self.rate = rate
        self.divid = divid
        self.vol = vol

    def gbm_paths(self, time=252, rand=[]):
        if not len(rand):
            rand = rng.standard_normal((time, nr_paths))
        paths = np.ones((1+time, nr_paths)) * self.spot
        for i in range(time):
            paths[i+1] = paths[i] * np.exp((self.rate - self.divid) * dt - \
                self.vol**2 * dt / 2 + self.vol * math.sqrt(dt) * rand[i])
        return paths

    def gbm_tree(self, time=252):
        self.up = math.exp(self.vol * math.sqrt(time * dt / nr_steps))
        self.down = 1/self.up
        self.pr = (math.exp((self.rate - self.divid) * time * dt / nr_steps) - \
            self.down) / (self.up - self.down)
        tree = [[self.spot] * (i+1) for i in range(nr_steps+1)]
        for i in range(1, nr_steps+1):
            for j in range(i+1):
                tree[i][j] *= self.up**(i-j) * self.down**j
        return tree
