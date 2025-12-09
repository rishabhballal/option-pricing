import math
import numpy as np

rng = np.random.default_rng()
count = 10**5
_dt = 1/252

class Stock:
    def __init__(self, spot=100, rate=0.05, divid=0.03, vol=0.10):
        self.spot = spot
        self.rate = rate
        self.divid = divid
        self.vol = vol

    def gbm_paths(self, time=252, rand=[]):
        if not len(rand):
            rand = rng.standard_normal((time, count))
        paths = np.ones((1+time, count)) * self.spot
        for i in range(time):
            paths[i+1] = paths[i] * np.exp((self.rate - self.divid) * _dt - \
                self.vol**2 * _dt / 2 + self.vol * math.sqrt(_dt) * rand[i])
        return paths

class _OptionGreeks:
    def __init__(self, option):
        self.option = option

    def delta(self, rand):
        epsilon = 0.01
        self.option.stock.spot += epsilon
        price_eps = self.option.price(rand)
        self.option.stock.spot -= epsilon
        return (price_eps - self.option.price(rand)) / epsilon

    def gamma(self, rand):
        epsilon = 0.01
        self.option.stock.spot += epsilon
        delta_eps = self.option.delta(rand)
        self.option.stock.spot -= epsilon
        return (delta_eps - self.option.delta(rand)) / epsilon

    def vega(self, rand):
        epsilon = 0.0001
        self.option.stock.vol += epsilon
        price_eps = self.option.price(rand)
        self.option.stock.vol -= epsilon
        return (price_eps - self.option.price(rand)) / epsilon

    def rho(self, rand):
        epsilon = 0.0001
        self.option.stock.rate += epsilon
        price_eps = self.option.price(rand)
        self.option.stock.rate -= epsilon
        return (price_eps - self.option.price(rand)) / epsilon

    def theta(self, rand):
        try:
            self.option.expiry -= 1
            price_eps = self.option.price(rand)
            self.option.expiry += 1
        except AttributeError:
            self.option.times = [x-1 for x in self.option.times]
            price_eps = self.option.price(rand)
            self.option.times = [x+1 for x in self.option.times]
        return (price_eps - self.option.price(rand)) / _dt

class PathIndependentEuropeanOption:
    def __init__(self, stock, expiry, payoff):
        self.stock = stock
        self.expiry = expiry
        self.payoff = lambda S: [payoff(x) for x in S]
        self.greeks = _OptionGreeks(self)

    def _random_seed(func):
        def wrapper(self, rand=[]):
            if not len(rand):
                rand = rng.standard_normal((self.expiry, count))
            return func(self, rand)
        return wrapper

    @_random_seed
    def price(self, rand):
        paths = self.stock.gbm_paths(self.expiry, rand)
        return math.exp(-self.stock.rate * self.expiry * _dt) * \
            np.mean(self.payoff(paths[-1]))

    @_random_seed
    def delta(self, rand):
        return self.greeks.delta(rand)

    @_random_seed
    def gamma(self, rand):
        return self.greeks.gamma(rand)

    @_random_seed
    def vega(self, rand):
        return self.greeks.vega(rand)

    @_random_seed
    def rho(self, rand):
        return self.greeks.rho(rand)

    @_random_seed
    def theta(self, rand):
        return self.greeks.theta(rand)

class PathDependentEuropeanOption:
    def __init__(self, stock, times, payoff):
        self.stock = stock
        self.times = times
        self.payoff = lambda S: [payoff(S[:,i]) for i in range(len(S[-1]))]
        self.greeks = _OptionGreeks(self)

    def _random_seed(func):
        def wrapper(self, rand=[]):
            if not len(rand):
                rand = rng.standard_normal((self.times[-1], count))
            return func(self, rand)
        return wrapper

    @_random_seed
    def price(self, rand):
        paths = self.stock.gbm_paths(self.times[-1], rand)
        for i in reversed(range(self.times[-1] + 1)):
            if i not in self.times:
                paths = np.delete(paths, i, axis=0)
        return math.exp(-self.stock.rate * self.times[-1] * _dt) * \
            np.mean(self.payoff(paths))

    @_random_seed
    def delta(self, rand):
        return self.greeks.delta(rand)

    @_random_seed
    def gamma(self, rand):
        return self.greeks.gamma(rand)

    @_random_seed
    def vega(self, rand):
        return self.greeks.vega(rand)

    @_random_seed
    def rho(self, rand):
        return self.greeks.rho(rand)

    @_random_seed
    def theta(self, rand):
        return self.greeks.theta(rand)
