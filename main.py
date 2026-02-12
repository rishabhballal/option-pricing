import option_pricing_engine as ope

# example of a stock
stock = ope.stock.GeometricBrownianMotion(
    spot=100,
    rate=0.05,
    divid=0.00,
    vol=0.25
)

# example of a European vanilla put option
option1 = ope.option.VanillaPut(
    stock=stock,
    expiry=252,
    strike=95
)
# example of an American vanilla put option
option2 = ope.option.PathIndependentOption(
    stock=stock,
    expiry=252,
    payoff=lambda spot: max(95 - spot, 0),
    ex_times=range(252)
)
# example of an arithmetic Asian put option
option3 = ope.option.EuropeanOption(
    stock=stock,
    expiry=252,
    payoff=lambda path: max(95 - sum(path)/len(path), 0),
    path_times=[63, 126, 189, 252]
)

print(f'Prices: {option1.price()}, {option2.price()}, {option3.price()}')
