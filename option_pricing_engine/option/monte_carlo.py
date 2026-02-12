import math
import numpy as np
from .. import rng, stock

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
                rand = rng.standard_normal((self.expiry + 1, stock.nr_paths))
            return func(self, rand)
        return wrapper

    @_random_seed
    def price(self, rand):
        paths = self.stock.paths(self.expiry, rand)
        reduced_paths = np.array([paths[t] for t in self.path_times])
        return math.exp(-self.stock.rate * self.expiry * stock.dt) * \
            np.mean(self.payoff(reduced_paths))

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
        return (price_1 - price_2) / (2 * stock.dt)

class LookbackCall(EuropeanOption):
    def __init__(self, stock, expiry, strike, path_times):
        if strike:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path: max(np.max(path) - strike, 0),
                path_times=path_times
            )
        else:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path: path[-1] - np.min(path),
                path_times=path_times
            )

class LookbackPut(EuropeanOption):
    def __init__(self, stock, expiry, strike, path_times):
        if strike:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path: max(strike - np.min(path), 0),
                path_times=path_times
            )
        else:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path: np.max(path) - path[-1],
                path_times=path_times
            )

class ArithmeticAsianCall(EuropeanOption):
    def __init__(self, stock, expiry, strike, path_times):
        if strike:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    max(np.sum(path) / np.size(path) - strike, 0),
                path_times=path_times
            )
        else:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    path[-1] - max(np.sum(path) / np.size(path), 0),
                path_times=path_times
            )

class ArithmeticAsianPut(EuropeanOption):
    def __init__(self, stock, expiry, strike, path_times):
        if strike:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    max(strike - np.sum(path) / np.size(path), 0),
                path_times=path_times
            )
        else:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    max(np.sum(path) / np.size(path) - path[-1], 0),
                path_times=path_times
            )

class GeometricAsianCall(EuropeanOption):
    def __init__(self, stock, expiry, strike, path_times):
        if strike:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    max(np.prod(path)**(1/np.size(path)) - strike, 0),
                path_times=path_times
            )
        else:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    path[-1] - max(np.prod(path)**(1/np.size(path)), 0),
                path_times=path_times
            )

class GeometricAsianPut(EuropeanOption):
    def __init__(self, stock, expiry, strike, path_times):
        if strike:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    max(strike - np.prod(path)**(1/np.size(path)), 0),
                path_times=path_times
            )
        else:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    max(np.prod(path)**(1/np.size(path)) - path[-1], 0),
                path_times=path_times
            )

class DiscreteBarrierKnockOutCall(EuropeanOption):
    def __init__(self, stock, expiry, strike, barrier, path_times):
        if stock.spot > barrier:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    max(path[-1] - strike, 0) if np.all(path > barrier) else 0,
                path_times=path_times
            )
        else:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    max(path[-1] - strike, 0) if np.all(path < barrier) else 0,
                path_times=path_times
            )

class DiscreteBarrierKnockOutPut(EuropeanOption):
    def __init__(self, stock, expiry, strike, barrier, path_times):
        if stock.spot > barrier:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    max(strike - path[-1], 0) if np.all(path > barrier) else 0,
                path_times=path_times
            )
        else:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    max(strike - path[-1], 0) if np.all(path < barrier) else 0,
                path_times=path_times
            )

class DiscreteBarrierKnockInCall(EuropeanOption):
    def __init__(self, stock, expiry, strike, barrier, path_times):
        if stock.spot > barrier:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    max(path[-1] - strike, 0) if np.any(path < barrier) else 0,
                path_times=path_times
            )
        else:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    max(path[-1] - strike, 0) if np.any(path > barrier) else 0,
                path_times=path_times
            )

class DiscreteBarrierKnockInPut(EuropeanOption):
    def __init__(self, stock, expiry, strike, barrier, path_times):
        if stock.spot > barrier:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    max(strike - path[-1], 0) if np.any(path < barrier) else 0,
                path_times=path_times
            )
        else:
            super().__init__(
                stock=stock,
                expiry=expiry,
                payoff=lambda path:
                    max(strike - path[-1], 0) if np.any(path > barrier) else 0,
                path_times=path_times
            )
