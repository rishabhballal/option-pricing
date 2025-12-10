import math
import numpy as np
import stocks

class PathIndependentEuropeanOption:
    def __init__(self, stock, expiry, payoff):
        self.stock = stock
        self.expiry = expiry
        self.payoff = lambda S: [payoff(x) for x in S]
        self.greeks = _OptionGreeks(self)

    def _random_seed(func):
        def wrapper(self, rand=[]):
            if not len(rand):
                rand = stocks.rng.standard_normal(
                    (self.expiry + 1, stocks.nr_paths))
            return func(self, rand)
        return wrapper

    @_random_seed
    def price(self, rand):
        paths = self.stock.gbm_paths(self.expiry, rand)
        return math.exp(-self.stock.rate * self.expiry * stocks.dt) * \
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
                rand = stocks.rng.standard_normal(
                    (self.times[-1] + 1, stocks.nr_paths))
            return func(self, rand)
        return wrapper

    @_random_seed
    def price(self, rand):
        paths = self.stock.gbm_paths(self.times[-1], rand)
        for i in reversed(range(self.times[-1] + 1)):
            if i not in self.times:
                paths = np.delete(paths, i, axis=0)
        return math.exp(-self.stock.rate * self.times[-1] * stocks.dt) * \
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

class _OptionGreeks:
    def __init__(self, option):
        self.option = option

    def delta(self, rand):
        epsilon = 0.01
        self.option.stock.spot += epsilon
        price_1 = self.option.price(rand)
        self.option.stock.spot -= 2 * epsilon
        price_2 = self.option.price(rand)
        self.option.stock.spot += epsilon
        return (price_1 - price_2) / (2 * epsilon)

    def gamma(self, rand):
        epsilon = 0.01
        self.option.stock.spot += epsilon
        delta_1 = self.option.delta(rand)
        self.option.stock.spot -= 2 * epsilon
        delta_2 = self.option.delta(rand)
        self.option.stock.spot += epsilon
        return (delta_1 - delta_2) / (2 * epsilon)

    def vega(self, rand):
        epsilon = 0.0001
        self.option.stock.vol += epsilon
        price_1 = self.option.price(rand)
        self.option.stock.vol -= 2 * epsilon
        price_2 = self.option.price(rand)
        self.option.stock.vol += epsilon
        return (price_1 - price_2) / (2 * epsilon)

    def rho(self, rand):
        epsilon = 0.0001
        self.option.stock.rate += epsilon
        price_1 = self.option.price(rand)
        self.option.stock.rate -= 2 * epsilon
        price_2 = self.option.price(rand)
        self.option.stock.rate += epsilon
        return (price_1 - price_2) / (2 * epsilon)

    def theta(self, rand):
        try:
            self.option.expiry -= 1
            price_1 = self.option.price(rand)
            self.option.expiry += 2
            price_2 = self.option.price(rand)
            self.option.expiry -= 1
        except AttributeError:
            self.option.times = [x-1 for x in self.option.times]
            price_1 = self.option.price(rand)
            self.option.times = [x+2 for x in self.option.times]
            price_2 = self.option.price(rand)
            self.option.times = [x-1 for x in self.option.times]
        return (price_1 - price_2) / (2 * stocks.dt)
