# Option pricing engine

This project calculates the prices and Greeks of vanilla options and two types of exotic options&mdash;those that have path-independent payoffs and might allow early exercise, and those that might have path-dependent payoffs but do not allow early exercise. Black-Scholes-Merton formulae, Cox-Ross-Rubinstein binomial trees, and Monte Carlo simulations serve as the conventional pricing mechanisms and are accordingly implemented here.

----

The first step is to define the underlying stock by providing its spot price, interest rate, dividend rate, and volatility.

```python
# main.py
import option_pricing_engine as ope

stock = ope.stock.GeometricBrownianMotion(
    spot=100,
    rate=0.05,
    divid=0.00,
    vol=0.25
)
```

The next step is to define the option.

* Black-Scholes-Merton formulae &ndash; for European vanilla options.

    Instantiate the `VanillaCall` or `VanillaPut` class by passing three arguments: the underlying stock, the number of days to expiry, and the strike price.

    ```python
    # main.py
    option1 = ope.option.VanillaPut(
        stock=stock,
        expiry=252,
        strike=95
    )
    ```

* Binomial trees &ndash; for European, American, or Bermudan options with path-independent payoffs.

    Instantiate the `PathIndependentOption` class by passing four arguments: the underlying stock, the number of days to expiry, the payoff function, and the list of days on which exercise is allowed.

    ```python
    # main.py
    option2 = ope.option.PathIndependentOption(
        stock=stock,
        expiry=252,
        payoff=lambda spot: max(95 - spot, 0),
        ex_times=range(252)
    )
    ```

    This is an American vanilla put option struck at 95. The last argument is optional and will default to the European case.

* Monte-Carlo simulations &ndash; for European options with path-dependent payoffs.

    Instantiate the `EuropeanOption` class by passing four arguments: the underlying, the number of days to expiry, the payoff function, and the list of days relevant to the payoff.

    ```python
    # main.py
    option3 = ope.option.EuropeanOption(
        stock=stock,
        expiry=252,
        payoff=lambda path: max(95 - sum(path)/len(path), 0),
        path_times=[63, 126, 189, 252]
    )
    ```

    This is an arithmetic Asian put option struck at 95. Note that the argument `path` of the payoff function will be a NumPy array of the stock prices at the times given in `path_times`. The last argument is optional and will default to the path-independent case. However, when it comes to options with path-independent payoffs, the binomial trees approach is much more efficient.

To output the price or a Greek (Delta, Gamma, Vega, Rho, Theta) of any instantiated option, simply call its identically-named method; _e.g._ `option1.price()`, `option2.delta()`, `option3.vega()`.

----

A few options have been predefined using this engine.

Option                      | Required arguments
--------------------------- | --------------------------------------------
Vanilla                     | `stock`, `expiry`, `strike`
Digital                     | `stock`, `expiry`, `strike`
Power                       | `stock`, `expiry`, `strike`, `power`
Straddle                    | `stock`, `expiry`, `strike`
American vanilla            | `stock`, `expiry`, `strike`
American digital            | `stock`, `expiry`, `strike`
American powers             | `stock`, `expiry`, `strike`, `power`
American straddle           | `stock`, `expiry`, `strike`
Bermudan vanilla            | `stock`, `expiry`, `strike`, `ex_times`
Bermudan digital            | `stock`, `expiry`, `strike`, `ex_times`
Bermudan power              | `stock`, `expiry`, `strike`, `ex_times`
Bermudan straddle           | `stock`, `expiry`, `strike`, `ex_times`
Lookback                    | `stock`, `expiry`, `strike`, `path_times`
Arithmetic Asian            | `stock`, `expiry`, `strike`, `path_times`
Geometric Asian             | `stock`, `expiry`, `strike`, `path_times`
Discrete barrier knock-out  | `stock`, `expiry`, `strike`, `barrier`, `path_times`
Discrete barrier knock-in   | `stock`, `expiry`, `strike`, `barrier`, `path_times`

Note that lookbacks, arithmetic Asians, and geometric Asians will have fixed or floating strikes depending on whether the `strike` argument is non-zero or zero respectively. Now, `option2` and `option3` can be rewritten semantically.

```python
# main.py
option2 = ope.option.AmericanVanillaPut(
    stock=stock,
    expiry=252,
    strike=95
)

option3 = ope.option.ArithmeticAsianPut(
    stock=stock,
    expiry=252,
    strike=95,
    path_times=[63, 126, 189, 252]
)
```
