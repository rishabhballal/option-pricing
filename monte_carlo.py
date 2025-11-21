import math
import numpy as np

rng = np.random.default_rng()

class MonteCarlo:
    def __init__(self, S_0, t, r, d, v, payoff):
        self.S_0 = S_0
        self.t = t
        self.r = r
        self.d = d
        self.v = v
        self.payoff = payoff

    def asset_singlestep(self, count):
        return [self.S_0*math.exp((self.r - self.d)*self.t - \
            0.5*(self.v**2)*self.t + self.v*math.sqrt(self.t)*x) \
            for x in rng.standard_normal(count)]

    def asset_multistep(self, count, N):
        S_t = np.ones(count)*self.S_0
        for i in range(1,N+1):
            S_t *= (1 + (self.r - self.d)*self.t/N + \
                self.v*math.sqrt(self.t/N) * rng.standard_normal(count))
        return S_t

    def price(self, count, N=1):
        S_t = self.asset_singlestep(count) if N == 1 \
            else self.asset_multistep(count, N)
        return np.mean([math.exp(-self.r*self.t) * self.payoff(x) for x in S_t])
