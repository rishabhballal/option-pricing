import math
import numpy as np
from scipy.stats import norm

# S - spot, K - strike, t - time to expiry, r - short rate
# d - dividend rate, v - volatility, F - forward price

discount = lambda t,r: math.exp(-r*t)
F = lambda S,t,r,d: S*math.exp((r-d)*t)
forward_contract = lambda S,K,t,r,d : discount(t,r)*(F(S,t,r,d) - K)

def auxiliary_vars(func):
    def wrapper(S,K,t,r,d,v):
        d1 = (math.log(S/K) + (r-d+0.5*v**2)*t)/(v*math.sqrt(t))
        d2 = d1 - v*math.sqrt(t)
        return func(S,K,t,r,d,v,d1,d2)
    return wrapper


# vanilla options

vanilla_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    discount(t,r)*(F(S,t,r,d)*norm.cdf(d1) - K*norm.cdf(d2)))

vanilla_put = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    discount(t,r)*(-F(S,t,r,d)*norm.cdf(-d1) + K*norm.cdf(-d2)))
    # vanilla_call(S,K,t,r,d,v) - forward_contract(S,K,t,r,d)

digital_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    discount(t,r)*norm.cdf(d2))
    # (vanilla_call(S,K-eps,t,r,d,v) - vanilla_call(S,K+eps,t,r,d,v))/(2*eps)

digital_put = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    discount(t,r)*norm.cdf(-d2))
    # discount(t,r) - digital_call(S,K,t,r,d,v)


# Greeks of a vanilla call option

n = lambda x: math.exp(-x**2/2)/math.sqrt(2*math.pi)

delta_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    discount(t,r)*F(1,t,r,d)*norm.cdf(d1))

gamma_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    (discount(t,r)*F(1,t,r,d)*n(d1))/(S*v*math.sqrt(t)))

vega_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    discount(t,r)*F(S,t,r,d)*n(d1))

rho_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    K*discount(t,r)*t*norm.cdf(d2))

theta_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    discount(t,r)*(F(S,t,r,d)*d*norm.cdf(d1) - K*r*norm.cdf(d2) - \
        F(S,t,r,d)*v*n(d1)/(2*math.sqrt(t))))


# Greeks of a vanilla put option

delta_put = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    -discount(t,r)*F(1,t,r,d)*norm.cdf(-d1))
    # delta_call(S,K,t,r,d,v) - math.exp(-d*t)

gamma_put = gamma_call

vega_put = vega_call

rho_put = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    -K*discount(t,r)*t*norm.cdf(-d2))
    # rho_call(S,K,t,r,d,v) - K*discount(t,r)*t

theta_put = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    discount(t,r)*(-F(S,t,r,d)*d*norm.cdf(-d1) + K*r*norm.cdf(-d2) - \
        F(S,t,r,d)*v*n(d1)/(2*math.sqrt(t))))
    # theta_call(S,K,t,r,d,v) - discount(t,r)*(F(S,t,r,d)*d - K*r)
