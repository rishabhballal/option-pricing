import numpy as np

# European, Bermudan, American option payoffs
forward_contract = lambda K: (lambda S: S-K)

vanilla_call = lambda K: (lambda S: S-K if S>K else 0)
vanilla_put = lambda K: (lambda S: K-S if S<K else 0)

digital_call = lambda K: (lambda S: 1 if S>K else 0)
digital_put = lambda K: (lambda S: 1 if S<K else 0)

power_call = lambda K, l: (lambda S: (S-K)^l if S>K else 0)
power_put = lambda K, l: (lambda S: (K-S)^l if S<K else 0)

straddle = lambda K: (lambda S: S-K if S>K else K-S)

# Asian option payoffs
arithmetic_asian_call = lambda K: (lambda S: sum(S)/len(S) - K \
    if sum(S)/len(S) > K else 0)
arithmetic_asian_put = lambda K: (lambda S: K - sum(S)/len(S) \
    if sum(S)/len(S) < K else 0)

geometric_asian_call = lambda K: (lambda S: np.prod(S)**(1/len(S)) - K \
    if np.prod(S)**(1/len(S)) > K else 0)
geometric_asian_put = lambda K: (lambda S: K - np.prod(S)**(1/len(S)) \
    if np.prod(S)**(1/len(S)) < K else 0)
