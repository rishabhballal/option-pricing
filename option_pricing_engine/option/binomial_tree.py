import math
import numpy as np
from .. import stock

class PathIndependentOption:
    def __init__(self, stock, expiry, payoff, ex_times=[]):
        self.stock = stock
        self.expiry = expiry
        self.payoff = lambda S: [payoff(x) for x in S]
        self.ex_times = ex_times if ex_times else [expiry]

    def _trees(self):
        S = self.stock.tree(self.expiry)
        V = [self.payoff(x) for x in S]
        for i in reversed(range(stock.nr_steps)):
            if int(i * self.expiry / stock.nr_steps) in self.ex_times:
                for j in range(i+1):
                    V[i][j] = max(
                        V[i][j],
                        math.exp(-self.stock.rate * self.expiry * stock.dt / \
                        stock.nr_steps) * (self.stock.pr_ * V[i+1][j] + \
                        (1 - self.stock.pr_) * V[i+1][j+1])
                    )
            else:
                for j in range(i+1):
                    V[i][j] = math.exp(-self.stock.rate * self.expiry * \
                        stock.dt / stock.nr_steps) * (self.stock.pr_ * \
                        V[i+1][j] + (1 - self.stock.pr_) * V[i+1][j+1])
        return S, V

    def price(self):
        return self._trees()[1][0][0]

    def delta(self):
        S, V = self._trees()
        return (V[1][0] - V[1][1]) / (S[1][0] - S[1][1])

    def gamma(self):
        S, V = self._trees()
        return 2 * (((V[2][0] - V[2][1]) / (S[2][0] - S[2][1])) - \
            ((V[2][1] - V[2][2]) / (S[2][1] - S[2][2]))) / (S[2][0] - S[2][2])

    def vega(self):
        epsilon = 0.0001
        self.stock.vol += epsilon
        price_eps = self.price()
        self.stock.vol -= epsilon
        return (price_eps - self.price()) / epsilon

    def rho(self):
        epsilon = 0.0001
        self.stock.rate += epsilon
        price_eps = self.price()
        self.stock.rate -= epsilon
        return (price_eps - self.price()) / epsilon

    def theta(self):
        S, V = self._trees()
        return (V[2][1] - V[0][0]) / \
            (2 * self.expiry * stock.dt / stock.nr_steps)

class DigitalCall(PathIndependentOption):
    def __init__(self, stock, expiry, strike):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: 1 if spot > strike else 0
        )

class DigitalPut(PathIndependentOption):
    def __init__(self, stock, expiry, strike):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: 1 if spot < strike else 0
        )

class PowerCall(PathIndependentOption):
    def __init__(self, stock, expiry, strike, power):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: (spot - strike)**power if spot > strike else 0
        )

class PowerPut(PathIndependentOption):
    def __init__(self, stock, expiry, strike, power):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: (strike - spot)**power if spot < strike else 0
        )

class Straddle(PathIndependentOption):
    def __init__(self, stock, expiry, strike):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: max(spot - strike, strike - spot)
        )

class AmericanVanillaCall(PathIndependentOption):
    def __init__(self, stock, expiry, strike):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: max(spot - strike, 0),
            ex_times=range(expiry)
        )

class AmericanVanillaPut(PathIndependentOption):
    def __init__(self, stock, expiry, strike):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: max(strike - spot, 0),
            ex_times=range(expiry)
        )

class AmericanDigitalCall(PathIndependentOption):
    def __init__(self, stock, expiry, strike):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: 1 if spot > strike else 0,
            ex_times=range(expiry)
        )

class AmericanDigitalPut(PathIndependentOption):
    def __init__(self, stock, expiry, strike):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: 1 if spot < strike else 0,
            ex_times=range(expiry)
        )

class AmericanPowerCall(PathIndependentOption):
    def __init__(self, stock, expiry, strike, power):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: (spot - strike)**power if spot > strike else 0,
            ex_times=range(expiry)
        )

class AmericanPowerPut(PathIndependentOption):
    def __init__(self, stock, expiry, strike, power):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: (strike - spot)**power if spot < strike else 0,
            ex_times=range(expiry)
        )

class AmericanStraddle(PathIndependentOption):
    def __init__(self, stock, expiry, strike):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: max(spot - strike, strike - spot),
            ex_times=range(expiry)
        )

class BermudanVanillaCall(PathIndependentOption):
    def __init__(self, stock, expiry, strike, ex_times):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: max(spot - strike, 0),
            ex_times=ex_times
        )

class BermudanVanillaPut(PathIndependentOption):
    def __init__(self, stock, expiry, strike, ex_times):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: max(strike - spot, 0),
            ex_times=ex_times
        )

class BermudanDigitalCall(PathIndependentOption):
    def __init__(self, stock, expiry, strike, ex_times):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: 1 if spot > strike else 0,
            ex_times=ex_times
        )

class BermudanDigitalPut(PathIndependentOption):
    def __init__(self, stock, expiry, strike, ex_times):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: 1 if spot < strike else 0,
            ex_times=ex_times
        )

class BermudanPowerCall(PathIndependentOption):
    def __init__(self, stock, expiry, strike, power, ex_times):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: (spot - strike)**power if spot > strike else 0,
            ex_times=ex_times
        )

class BermudanPowerPut(PathIndependentOption):
    def __init__(self, stock, expiry, strike, power, ex_times):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: (strike - spot)**power if spot < strike else 0,
            ex_times=ex_times
        )

class BermudanStraddle(PathIndependentOption):
    def __init__(self, stock, expiry, strike, ex_times):
        super().__init__(
            stock=stock,
            expiry=expiry,
            payoff=lambda spot: max(spot - strike, strike - spot),
            ex_times=ex_times
        )
