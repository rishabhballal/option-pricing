import math
import numpy as np
from scipy.stats import norm

# S - spot, K - strike, t - time to expiry, r - short rate
# d - dividend rate, v - volatility, F - forward price

zc_bond = lambda t,r: math.exp(-r*t)
F = lambda P,t,r,d: P*math.exp((r-d)*t)
forward_contract = lambda S,K,t,r,d : zc_bond(t,r)*(F(S,t,r,d) - K)

def auxiliary_vars(func):
    def wrapper(S,K,t,r,d,v):
        d1 = (math.log(S/K) + (r-d+0.5*v**2)*t)/(v*math.sqrt(t))
        d2 = d1 - v*math.sqrt(t)
        return func(S,K,t,r,d,v,d1,d2)
    return wrapper

vanilla_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    zc_bond(t,r)*(F(S,t,r,d)*norm.cdf(d1) - K*norm.cdf(d2)))

vanilla_put = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    zc_bond(t,r)*(-F(S,t,r,d)*norm.cdf(-d1) + K*norm.cdf(-d2)))
    # vanilla_call(S,K,t,r,d,v) - forward_contract(S,K,t,r,d)

digital_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    zc_bond(t,r)*norm.cdf(d2))

digital_put = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    zc_bond(t,r)*norm.cdf(-d2))
    # zc_bond(t,r) - digital_call(S,K,t,r,d,v)
