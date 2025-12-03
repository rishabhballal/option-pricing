import math
import numpy as np

class Stock:
    def __init__(self, spot=100, rate=0.05, divid=0.03, vol=0.10):
        self.spot = spot
        self.rate = rate
        self.divid = divid
        self.vol = vol

    def geom_brownian(self, time=1, steps=1000):
        self.up = math.exp(self.vol * math.sqrt(time / steps))
        self.down = 1/self.up
        self.pr = (math.exp((self.rate - self.divid) * time / steps) - \
            self.down) / (self.up - self.down)
        tree = [[self.spot] * (i+1) for i in range(steps+1)]
        for i in range(1, steps+1):
            for j in range(i+1):
                tree[i][j] *= self.up**(i-j) * self.down**j
        return tree

class EuropeanOption:
    def __init__(self, stock, expiry, payoff):
        self.stock = stock
        self.expiry = expiry
        self.payoff = lambda S: [payoff(x) for x in S]

    def _trees(self, steps):
        S = self.stock.geom_brownian(self.expiry, steps)
        V = [self.payoff(x) for x in S]
        for i in reversed(range(steps)):
            for j in range(i+1):
                V[i][j] = math.exp(-self.stock.rate * self.expiry / steps) * \
                    (self.stock.pr * V[i+1][j] + (1 - self.stock.pr) * \
                    V[i+1][j+1])
        return S, V

    price = lambda self, steps=1000: self._trees(steps)[1][0][0]

    def delta(self, steps=1000):
        S, V = self._trees(steps)
        return (V[1][0] - V[1][1]) / (S[1][0] - S[1][1])

    def gamma(self, steps=1000):
        S, V = self._trees(steps)
        return 2 * (((V[2][0] - V[2][1]) / (S[2][0] - S[2][1])) - \
            ((V[2][1] - V[2][2]) / (S[2][1] - S[2][2]))) / (S[2][0] - S[2][2])

    def vega(self, steps=1000):
        epsilon = 0.0001
        self.stock.vol += epsilon
        price_eps = self.price(steps)
        self.stock.vol -= epsilon
        return (price_eps - self.price(steps)) / epsilon

    def rho(self, steps=1000):
        epsilon = 0.0001
        self.stock.rate += epsilon
        price_eps = self.price(steps)
        self.stock.rate -= epsilon
        return (price_eps - self.price(steps)) / epsilon

    def theta(self, steps=1000):
        S, V = self._trees(steps)
        return (V[2][1] - V[0][0]) / (2 * self.expiry / steps)

class AmericanOption:
    def __init__(self, stock, expiry, payoff):
        self.stock = stock
        self.expiry = expiry
        self.payoff = lambda S: [payoff(x) for x in S]

    def price(self, steps=1000):
        S = self.stock.geom_brownian(self.expiry, steps)
        V = [self.payoff(x) for x in S]
        for i in reversed(range(steps)):
            for j in range(i+1):
                V[i][j] = max(V[i][j], math.exp(-self.stock.rate * \
                    self.expiry / steps) * (self.stock.pr * V[i+1][j] + \
                    (1 - self.stock.pr) * V[i+1][j+1]))
        return V[0][0]

class BermudanOption:
    def __init__(self, stock, times, payoff):
        self.stock = stock
        self.times = times
        self.payoff = lambda S: [payoff(x) for x in S]

    def price(self, steps_=100):
        total_steps = steps_ * len(self.times)
        dt = self.times[-1] / total_steps
        S = self.stock.geom_brownian(self.times[-1], total_steps)
        V = [self.payoff(x) for x in S]
        for i in reversed(range(total_steps)):
            if i*dt in self.times:
                for j in range(i+1):
                    V[i][j] = max(V[i][j], math.exp(-self.stock.rate * \
                        dt) * (self.stock.pr * V[i+1][j] + \
                        (1 - self.stock.pr) * V[i+1][j+1]))
            else:
                for j in range(i+1):
                    V[i][j] = math.exp(-self.stock.rate * dt) * \
                        (self.stock.pr * V[i+1][j] + (1 - self.stock.pr) * \
                        V[i+1][j+1])
        return V[0][0]
