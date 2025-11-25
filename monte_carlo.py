import math
import numpy as np

rng = np.random.default_rng()

class Stock:
    def __init__(self, count=10**6, spot=100, rate=0.05, divid=0.03, vol=0.10):
        self.count = count
        self.spot = spot
        self.rate = rate
        self.divid = divid
        self.vol = vol

    def geom_brownian(self, time=1, steps=1, rand=[]):
        if not len(rand):
            rand = rng.standard_normal((steps, self.count))
        S_t = np.ones((steps+1, self.count)) * self.spot
        if not steps-1:
            S_t[-1] = [self.spot * math.exp((self.rate - self.divid) * time - \
                0.5 * (self.vol**2) * time + self.vol * math.sqrt(time) * x) \
                for x in rand[-1]]
        else:
            for i in range(steps):
                S_t[i+1] = S_t[i] * (1 + (self.rate - self.divid) * time / \
                steps + self.vol * math.sqrt(time / steps) * rand[i])
        return S_t

class EuropeanOption:
    def __init__(self, stock, expiry, payoff):
        self.stock = stock
        self.expiry = expiry
        self.payoff = lambda S: np.array([payoff(x) for x in S])

    def _random_seed(func):
        def wrapper(self, steps=1, rand=[]):
            if not len(rand):
                rand = rng.standard_normal((steps, self.stock.count))
            return func(self, steps, rand)
        return wrapper

    @_random_seed
    def price(self, steps, rand):
        S_t = self.stock.geom_brownian(self.expiry, steps, rand)[-1]
        return math.exp(-self.stock.rate*self.expiry) * \
            np.mean(self.payoff(S_t))

    @_random_seed
    def delta(self, steps, rand):
        epsilon = 0.01
        self.stock.spot += epsilon
        price_eps = self.price(steps, rand)
        self.stock.spot -= epsilon
        return (price_eps - self.price(steps, rand))/epsilon

    @_random_seed
    def gamma(self, steps, rand):
        epsilon = 0.01
        self.stock.spot += epsilon
        delta_eps = self.delta(steps, rand)
        self.stock.spot -= epsilon
        return (delta_eps - self.delta(steps, rand))/epsilon

    @_random_seed
    def vega(self, steps, rand):
        epsilon = 0.0001
        self.stock.vol += epsilon
        price_eps = self.price(steps, rand)
        self.stock.vol -= epsilon
        return (price_eps - self.price(steps, rand))/epsilon

    @_random_seed
    def rho(self, steps, rand):
        epsilon = 0.0001
        self.stock.rate += epsilon
        price_eps = self.price(steps, rand)
        self.stock.rate -= epsilon
        return (price_eps - self.price(steps, rand))/epsilon

    @_random_seed
    def theta(self, steps, rand):
        epsilon = 0.01
        self.expiry += epsilon
        price_eps = self.price(steps, rand)
        self.expiry -= epsilon
        return -(price_eps - self.price(steps, rand))/epsilon
