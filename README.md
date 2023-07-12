# Algorithmic-Trading
Hidden markov model regime switching and OPTICS-clustering pairs trading algorithms, built in Python to be used on QuantConnect website

# HMM_regime_switching.py: 
Inspired by a 2020 paper by Matthew Wang (https://www.mdpi.com/1911-8074/13/12/311), this algorithm trains a hidden markov model to analyze historic SPY returns and volatility and categorize current market into one of three regimes (output states)--bull, bear, or neither. It then employs one of two trading models depending on the HMM output. In a "bull" market, it employs a long-only value model, and in a "bear" market, it employs a modified version of the Fama-French 3 factor model. Like Wang's, this algorithm also uses a high-pass filter to prevent overly-frequent and costly shifts between market regimes.

The code Wang provides is unable to reproduce anything approaching his recorded results when backtested over the same time period. Significant changes were made to attempt to improve the algorithm while keeping the main components in place. In Wang's original code, the HMM would be trained every day and return an output state, but this would be largely ignored by the algorithm and the factor model used would not reflect the opinion of the markov model. This is remedied here.

Furthermore, this code also incorporates stock sentiment data from Tiingo, discarding a stock from the portfolio if recent public sentiment regarding the underlying company is sufficiently distant from the algorithm's opinion of the stock.

# HMM_mkt_orders.py: 
This is the same as above, but the factor models use a different (and in the end much more complicated) method to buy and sell shares, such that multiple strategies can be simultaneously run under one portfolio on QuantConnect. This algorithm uses the MarketOrder method instead of the SetHoldings method. This way the hmm-regime-switching strategy can buy and sell shares of a stock without potentially affecting shares of the same stock allocated toward a different strategy

# Pairs_trading.py:
This algorithm deploys a chain of techniques and tests to find pairs of stocks that are highly cointegrated, and then employs a simple pairs trading strategy based on the assumption of mean reversion. After filtering based on market cap, 180 day returns and other fundamental data are tracked. PCA is used to reduce dimensionality so that the OPTICS density-based clustering algorithm can be performed for stocks in each market sector. Within each cluster, cointegration and ADF tests are iteratively performed such that the most optimal pairs can be selected. Allocation to each selected pair is based on equal CTR, and trading is done with a triple-barrier risk management mechanism in place, utilizing stop loss thresholds, time barrier exits, and taking profits when the mean of each pair reverts by crossing zero.
