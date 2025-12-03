import math
import numpy as np

rng = np.random.default_rng()

class Stock:
    def __init__(self, spot=100, rate=0.05, divid=0.03, vol=0.10, count=10**4):
        self.count = count
        self.spot = spot
        self.rate = rate
        self.divid = divid
        self.vol = vol

    def geom_brownian(self, time=1, steps=10**2, rand=[]):
        if not len(rand):
            rand = rng.standard_normal((steps, self.count))
        S = np.ones((steps+1, self.count)) * self.spot
        if steps == 1:
            S[-1] = [self.spot * math.exp((self.rate - self.divid) * time - \
                0.5 * (self.vol**2) * time + self.vol * math.sqrt(time) * x) \
                for x in rand[-1]]
        else:
            for i in range(steps):
                S[i+1] = S[i] * (1 + (self.rate - self.divid) * time / \
                    steps + self.vol * math.sqrt(time / steps) * rand[i])
        return S

class EuropeanOption:
    def __init__(self, stock, expiry, payoff):
        self.stock = stock
        self.expiry = expiry
        self.payoff = lambda S: [payoff(x) for x in S]

    def _random_seed(func):
        def wrapper(self, steps=10**3, rand=[]):
            if not len(rand):
                rand = rng.standard_normal((steps, self.stock.count))
            return func(self, steps, rand)
        return wrapper

    @_random_seed
    def price(self, steps, rand):
        S_T = self.stock.geom_brownian(self.expiry, steps, rand)[-1]
        return math.exp(-self.stock.rate * self.expiry) * \
            np.mean(self.payoff(S_T))

    @_random_seed
    def delta(self, steps, rand):
        epsilon = 0.01
        self.stock.spot += epsilon
        price_eps = self.price(steps, rand)
        self.stock.spot -= epsilon
        return (price_eps - self.price(steps, rand)) / epsilon

    @_random_seed
    def gamma(self, steps, rand):
        epsilon = 0.01
        self.stock.spot += epsilon
        delta_eps = self.delta(steps, rand)
        self.stock.spot -= epsilon
        return (delta_eps - self.delta(steps, rand)) / epsilon

    @_random_seed
    def vega(self, steps, rand):
        epsilon = 0.0001
        self.stock.vol += epsilon
        price_eps = self.price(steps, rand)
        self.stock.vol -= epsilon
        return (price_eps - self.price(steps, rand)) / epsilon

    @_random_seed
    def rho(self, steps, rand):
        epsilon = 0.0001
        self.stock.rate += epsilon
        price_eps = self.price(steps, rand)
        self.stock.rate -= epsilon
        return (price_eps - self.price(steps, rand)) / epsilon

    @_random_seed
    def theta(self, steps, rand):
        epsilon = 0.01
        self.expiry -= epsilon
        price_eps = self.price(steps, rand)
        self.expiry += epsilon
        return (price_eps - self.price(steps, rand)) / epsilon

class AsianOption:
    def __init__(self, stock, times, payoff):
        self.stock = stock
        self.times = times
        self.payoff = lambda S: [payoff(S[:,i]) for i in range(stock.count)]

    def _random_seed(func):
        def wrapper(self, steps_=10**2, rand=[]):
            if not len(rand):
                rand = rng.standard_normal((steps_ * len(self.times), \
                    self.stock.count))
            return func(self, steps_, rand)
        return wrapper

    @_random_seed
    def price(self, steps_, rand):
        total_steps = steps_ * len(self.times)
        S = self.stock.geom_brownian(self.times[-1], total_steps, rand)
        for i in reversed(range(total_steps+1)):
            if i * self.times[-1] / total_steps not in self.times:
                S = np.delete(S, i, axis=0)
        return math.exp(-self.stock.rate * self.times[-1]) * \
            np.mean(self.payoff(S))

    @_random_seed
    def delta(self, steps_, rand):
        epsilon = 0.01
        self.stock.spot += epsilon
        price_eps = self.price(steps_, rand)
        self.stock.spot -= epsilon
        return (price_eps - self.price(steps_, rand)) / epsilon

    @_random_seed
    def gamma(self, steps_, rand):
        epsilon = 0.01
        self.stock.spot += epsilon
        delta_eps = self.delta(steps_, rand)
        self.stock.spot -= epsilon
        return (delta_eps - self.delta(steps_, rand)) / epsilon

    @_random_seed
    def vega(self, steps_, rand):
        epsilon = 0.0001
        self.stock.vol += epsilon
        price_eps = self.price(steps_, rand)
        self.stock.vol -= epsilon
        return (price_eps - self.price(steps_, rand)) / epsilon

    @_random_seed
    def rho(self, steps_, rand):
        epsilon = 0.0001
        self.stock.rate += epsilon
        price_eps = self.price(steps_, rand)
        self.stock.rate -= epsilon
        return (price_eps - self.price(steps_, rand)) / epsilon

    @_random_seed
    def theta(self, steps_, rand):
        epsilon = 0.01
        for i in range(len(self.times)):
            self.times[i] -= epsilon * (i+1)
        price_eps = self.price(steps_, rand)
        for i in range(len(self.times)):
            self.times[i] += epsilon * (i+1)
        return (price_eps - self.price(steps_, rand)) / epsilon

class LookbackOption:
    def __init__(self, stock, expiry, payoff):
        self.stock = stock
        self.expiry = expiry
        self.payoff = lambda S, S_min, S_max: [payoff(S[i], S_min[i], S_max[i]) for i in range(self.stock.count)]

    def _random_seed(func):
        def wrapper(self, steps=10**3, rand=[]):
            if not len(rand):
                rand = rng.standard_normal((steps, self.stock.count))
            return func(self, steps, rand)
        return wrapper

    @_random_seed
    def price(self, steps, rand):
        S = self.stock.geom_brownian(self.expiry, steps, rand)
        S_min = [min(S[:,i]) for i in range(self.stock.count)]
        S_max = [max(S[:,i]) for i in range(self.stock.count)]
        return math.exp(-self.stock.rate * self.expiry) * \
            np.mean(self.payoff(S[-1], S_min, S_max))
