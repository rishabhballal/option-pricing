import math
import numpy as np
import stocks

class EuropeanOption:
    def __init__(self, stock, expiry, payoff, path_times=[]):
        self.stock = stock
        self.expiry = expiry
        if path_times:
            self.payoff = lambda S: [payoff(S[:,i]) for i in range(len(S[-1]))]
            self.path_times = path_times
        else:
            self.payoff = lambda S: [payoff(x) for x in S[-1]]
            self.path_times = [expiry]

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
        for i in reversed(range(self.expiry + 1)):
            if i not in self.path_times:
                paths = np.delete(paths, i, axis=0)
        return math.exp(-self.stock.rate * self.expiry * stocks.dt) * \
            np.mean(self.payoff(paths))

    @_random_seed
    def delta(self, rand):
        epsilon = 0.01
        self.stock.spot += epsilon
        price_1 = self.price(rand)
        self.stock.spot -= 2 * epsilon
        price_2 = self.price(rand)
        self.stock.spot += epsilon
        return (price_1 - price_2) / (2 * epsilon)

    @_random_seed
    def gamma(self, rand):
        epsilon = 0.01
        self.stock.spot += epsilon
        delta_1 = self.delta(rand)
        self.stock.spot -= 2 * epsilon
        delta_2 = self.delta(rand)
        self.stock.spot += epsilon
        return (delta_1 - delta_2) / (2 * epsilon)

    @_random_seed
    def vega(self, rand):
        epsilon = 0.0001
        self.stock.vol += epsilon
        price_1 = self.price(rand)
        self.stock.vol -= 2 * epsilon
        price_2 = self.price(rand)
        self.stock.vol += epsilon
        return (price_1 - price_2) / (2 * epsilon)

    @_random_seed
    def rho(self, rand):
        epsilon = 0.0001
        self.stock.rate += epsilon
        price_1 = self.price(rand)
        self.stock.rate -= 2 * epsilon
        price_2 = self.price(rand)
        self.stock.rate += epsilon
        return (price_1 - price_2) / (2 * epsilon)

    @_random_seed
    def theta(self, rand):
        self.expiry -= 1
        self.path_times = [x-1 for x in self.path_times]
        price_1 = self.price(rand)
        self.expiry += 2
        self.path_times = [x+2 for x in self.path_times]
        price_2 = self.price(rand)
        self.expiry -= 1
        self.path_times = [x-1 for x in self.path_times]
        return (price_1 - price_2) / (2 * stocks.dt)
        # why does theta have such a large variance?!
