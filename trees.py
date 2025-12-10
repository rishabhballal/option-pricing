import math
import numpy as np
import stocks

class _Option:
    def __init__(self, stock, payoff):
        self.stock = stock
        self.payoff = lambda S: [payoff(x) for x in S]

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
        return (V[2][1] - V[0][0]) / (2 * self.expiry / stocks.steps)

class EuropeanOption(_Option):
    def __init__(self, stock, expiry, payoff):
        super().__init__(stock, payoff)
        self.expiry = expiry

    def _trees(self):
        S = self.stock.gbm_tree(self.expiry)
        V = [self.payoff(x) for x in S]
        for i in reversed(range(stocks.nr_steps)):
            for j in range(i+1):
                V[i][j] = math.exp(-self.stock.rate * self.expiry * \
                    stocks.dt / stocks.nr_steps) * (self.stock.pr * \
                    V[i+1][j] + (1 - self.stock.pr) * V[i+1][j+1])
        return S, V

class AmericanOption(_Option):
    def __init__(self, stock, expiry, payoff):
        super().__init__(stock, payoff)
        self.expiry = expiry

    def _trees(self):
        S = self.stock.gbm_tree(self.expiry)
        V = [self.payoff(x) for x in S]
        for i in reversed(range(stocks.nr_steps)):
            for j in range(i+1):
                V[i][j] = max(
                    V[i][j],
                    math.exp(-self.stock.rate * self.expiry * stocks.dt / \
                    stocks.nr_steps) * (self.stock.pr * V[i+1][j] + \
                    (1 - self.stock.pr) * V[i+1][j+1]))
        return S, V

class BermudanOption(_Option):
    def __init__(self, stock, times, payoff):
        super().__init__(stock, payoff)
        self.times = times

    def _trees(self, steps_=100):
        total_steps = steps_ * len(self.times)
        dt = self.times[-1] / total_steps
        S = self.stock.gbm_tree(self.times[-1], total_steps)
        V = [self.payoff(x) for x in S]
        for i in reversed(range(total_steps)):
            if i*dt in self.times:
                for j in range(i+1):
                    V[i][j] = max(
                        V[i][j],
                        math.exp(-self.stock.rate * dt) * (self.stock.pr * \
                        V[i+1][j] + (1 - self.stock.pr) * V[i+1][j+1]))
            else:
                for j in range(i+1):
                    V[i][j] = math.exp(-self.stock.rate * dt) * \
                        (self.stock.pr * V[i+1][j] + (1 - self.stock.pr) * \
                        V[i+1][j+1])
        return S, V
