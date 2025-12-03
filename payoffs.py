import numpy as np

# European, Bermudan, American option payoffs
def forward_contract(K):
    return (lambda S: S-K)

def vanilla_call(K):
    return (lambda S: S-K if S>K else 0)
def vanilla_put(K):
    return (lambda S: K-S if S<K else 0)

def digital_call(K):
    return (lambda S: 1 if S>K else 0)
def digital_put(K):
    return (lambda S: 1 if S<K else 0)

def power_call(K, l):
    return (lambda S: (S-K)^l if S>K else 0)
def power_put(K, l):
    return (lambda S: (K-S)^l if S<K else 0)

def straddle(K):
    return (lambda S: S-K if S>K else K-S)

# Asian option payoffs
def arithmetic_asian_call(K):
    return (lambda S: sum(S)/len(S) - K if sum(S)/len(S) > K else 0)
def arithmetic_asian_put(K):
    return (lambda S: K - sum(S)/len(S) if sum(S)/len(S) < K else 0)

def geometric_asian_call(K):
    return (lambda S: np.prod(S)**(1/len(S)) - K if np.prod(S)**(1/len(S)) > K \
        else 0)
def geometric_asian_put(K):
    return (lambda S: K - np.prod(S)**(1/len(S)) if np.prod(S)**(1/len(S)) < K \
        else 0)

# lookback option payoffs
def lookback_call_fixed(K):
    return (lambda S, S_min, S_max: S_max - K if S_max > K else 0)
def lookback_put_fixed(K):
    return (lambda S, S_min, S_max: K - S_min if S_min < K else 0)

def lookback_call_floating():
    return (lambda S, S_min, S_max: S - S_min)
def lookback_put_floating():
    return (lambda S, S_min, S_max: S_max - S)
