import math
import numpy as np

rng = np.random.default_rng()

class Stock:
    def __init__(self, spot=100, rate=0.05, divid=0.03, vol=0.10):
        self.spot = spot
        self.rate = rate
        self.divid = divid
        self.vol = vol

    def geom_brownian(self, time=1, steps=3):
        S = np.ones((steps+1, 2**steps)) * math.log(self.spot)
        for i in range(steps):
            for j in range(2**steps):
                if j % 2**(steps-i) < 2**(steps-i-1):
                    S[i+1,j] = S[i,j] + (self.rate - self.divid - 0.5 * \
                        self.vol**2) * time / steps + self.vol * \
                        math.sqrt(time / steps)
                else:
                    S[i+1,j] = S[i,j] + (self.rate - self.divid - 0.5 * \
                        self.vol**2) * time / steps - self.vol * \
                        math.sqrt(time / steps)
        return np.exp(S)

class EuropeanOption:
    def __init__(self, stock, expiry, payoff):
        self.stock = stock
        self.expiry = expiry
        self.payoff = lambda S: np.array([payoff(x) for x in S])

    def price(self, steps=18):
        S_T1 = self.stock.geom_brownian(self.expiry, steps)[-1]
        if steps==1:
            S_T2 = S_T1
        else:
            S_T2 = self.stock.geom_brownian(self.expiry, steps-1)[-1]
        return math.exp(-self.stock.rate*self.expiry) * \
            (np.mean(self.payoff(S_T1)) + np.mean(self.payoff(S_T2))) / 2

    def delta(self, steps=18):
        epsilon = 0.01
        self.stock.spot += epsilon
        price_eps = self.price(steps)
        self.stock.spot -= epsilon
        return (price_eps - self.price(steps)) / epsilon

    def gamma(self, steps=18):
        epsilon = 10
        # why does it only work with a large epsilon?
        self.stock.spot += epsilon
        price_plus = self.price(steps)
        self.stock.spot -= 2*epsilon
        price_minus = self.price(steps)
        self.stock.spot += epsilon
        return (price_plus - 2*self.price(steps) + price_minus) / epsilon**2

    def vega(self, steps=18):
        epsilon = 0.0001
        self.stock.vol += epsilon
        price_eps = self.price(steps)
        self.stock.vol -= epsilon
        return (price_eps - self.price(steps)) / epsilon

    def rho(self, steps=18):
        epsilon = 0.0001
        self.stock.rate += epsilon
        price_eps = self.price(steps)
        self.stock.rate -= epsilon
        return (price_eps - self.price(steps)) / epsilon

    def theta(self, steps=18):
        epsilon = 0.01
        self.expiry += epsilon
        price_eps = self.price(steps)
        self.expiry -= epsilon
        return -(price_eps - self.price(steps)) / epsilon
