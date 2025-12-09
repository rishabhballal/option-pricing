import numpy as np

# some payoffs for instantiating trees.EuropeanOption, trees.AmericanOption,
# trees.BermudanOption, monte_carlo.PathIndependentEuropeanOption

def forward_contract(strike):
    return (lambda spot: spot - strike)

def vanilla_call(strike):
    return (lambda spot: max(spot - strike, 0))
def vanilla_put(strike):
    return (lambda spot: max(strike - spot, 0))

def digital_call(strike):
    return (lambda spot: 1 if spot > strike else 0)
def digital_put(strike):
    return (lambda spot: 1 if spot < strike else 0)

def power_call(strike, power):
    return (lambda spot: (spot - strike)**power if spot > strike else 0)
def power_put(strike, power):
    return (lambda spot: (strike - spot)**power if spot < strike else 0)

def straddle(strike):
    return (lambda spot: max(spot - strike, strike - spot))


# some payoffs for instantiating monte_carlo.PathDependentEuropeanOption

def fixed_lookback_call(strike):
    return (lambda path: max(np.max(path) - strike, 0))
def fixed_lookback_put(strike):
    return (lambda path: max(strike - np.min(path), 0))

def floating_lookback_call():
    return (lambda path: path[-1] - np.min(path))
def floating_lookback_put():
    return (lambda path: np.max(path) - path[-1])

def arithmetic_asian_call(strike):
    return (lambda path: max(np.sum(path) / np.size(path) - strike, 0))
def arithmetic_asian_put(strike):
    return (lambda path: max(strike - np.sum(path) / np.size(path), 0))

def geometric_asian_call(strike):
    return (lambda path: max(np.prod(path)**(1 / np.size(path)) - strike, 0))
def geometric_asian_put(strike):
    return (lambda path: max(strike - np.prod(path)**(1 / np.size(path)), 0))

def discrete_down_and_out_call(strike, barrier):
    return (lambda path: max(path[-1] - strike, 0) \
        if np.all(path > barrier) else 0)
def discrete_down_and_out_put(strike, barrier):
    return (lambda path: max(strike - path[-1], 0) \
        if np.all(path > barrier) else 0)

def discrete_down_and_in_call(strike, barrier):
    return (lambda path: max(path[-1] - strike, 0) \
        if np.any(path < barrier) else 0)
def discrete_down_and_in_put(strike, barrier):
    return (lambda path: max(strike - path[-1], 0) \
        if np.any(path < barrier) else 0)

def discrete_up_and_out_call(strike, barrier):
    return (lambda path: max(path[-1] - strike, 0) \
        if np.all(path < barrier) else 0)
def discrete_up_and_out_put(strike, barrier):
    return (lambda path: max(strike - path[-1], 0) \
        if np.all(path < barrier) else 0)

def discrete_up_and_in_call(strike, barrier):
    return (lambda path: max(path[-1] - strike, 0) \
        if np.any(path > barrier) else 0)
def discrete_up_and_in_put(strike, barrier):
    return (lambda path: max(strike - path[-1], 0) \
        if np.any(path > barrier) else 0)
