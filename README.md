# Option pricing engine

This project is a work-in-progress. As of now, one can follow this simple guide to price vanillas and some standard exotic options. In `main.py`, first import the `stocks.py` module and instantiate its `Stock` class by providing the spot price, interest rate, dividend rate, and volatility.

```python
import stocks

stock = stocks.Stock(
    spot=100,
    rate=0.05,
    divid=0.00,
    vol=0.25
)
```

This object will serve as the underlying in this guide. Depending upon the features of the option in mind, one then selects an appropriate pricing method.

## 1. Black-Scholes-Merton formulae

Suitable for vanilla options without an early exercise feature, _i.e._ European vanilla options. Import the `black_scholes_merton.py` module and instantiate its `VanillaCall` or `VanillaPut` class by passing three arguments: the underlying, the time to expiry in days, and a strike price.

```python
import black_scholes_merton as bsm

option1 = bsm.VanillaPut(
    stock=stock,
    expiry=252,
    strike=95
)
```

## 2. Binomial trees

Suitable for options with a path-independent payoff and possibly an early exercise feature. Import the `trees.py` module and instantiate its `PathIndependentOption` class by passing four arguments: the underlying, the time to expiry in days, a payoff function, and the times at which exercise is allowed.

```python
import trees

option2 = trees.PathIndependentOption(
    stock=stock,
    expiry=252,
    payoff=lambda spot: max(95 - spot, 0),
    ex_times=range(252)
)
```

It is easy to see that `option2` is an American vanilla put option struck at 95. The last argument is optional; `ex_times=[]` by default, which corresponds to the European case. Setting `ex_times=[63, 126, 189, 252]` in `option2` would be an example of the Bermudan case.

The number of steps in the binomial tree can be adjusted with `stocks.nr_steps`.

## 3. Monte-Carlo simulations

Suitable for options with a path-independent or path-dependent payoff but no early exercise feature. Import the `monte_carlo.py` module and instantiate its `EuropeanOption` class by passing four arguments: the underlying, the time to expiry in days, a payoff function, and the times relevant to the payoff.

```python
import monte_carlo as mc

option3 = mc.EuropeanOption(
    stock=stock,
    expiry=252,
    payoff=lambda path: max(95 - sum(path)/len(path), 0),
    path_times=[63, 126, 189, 252]
)
```

From the payoff, `option3` is recognised as an arithmetic Asian put option struck at 95. Note that the argument `path` of the payoff function will be a NumPy array of stock prices at the times given in `path_times`. Once again, the last argument is optional; `path_times=[]` by default, which corresponds to the path-independent case. However, when it comes to options with path-independent payoffs, the binomial trees approach is much more efficient.

The number of paths in the Monte Carlo simulations can be adjusted with `stocks.nr_paths`.

## Payoff functions

Some standard payoff functions are already defined in `payoffs.py`, and fall into two categories.

* Path-independent: forward contract, vanilla call and put, digital call and put, power call and put, and straddles.
* Path-dependent: fixed and floating lookback calls and puts, arithmetic and geometric Asian calls and puts, and discrete barrier calls and puts.

On importing the `payoffs.py` module, its functions offer semantic ease when instantiating options. For example, the payoff arguments of `option2` and `option3` can be rewritten as `payoffs.vanilla_put(strike=95)` and `payoffs.arithmetic_asian_put(strike=95)` respectively.

## Price and Greeks

To output the price or a Greek (Delta, Gamma, Vega, Rho, Theta) of any instantiated option, simply call its identically-named method; _e.g._ `option1.price()`, `option2.delta()`, `option3.vega()`.
