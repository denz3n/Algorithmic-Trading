import operator
from math import ceil,floor
import pandas as pd
import scipy as scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hmmlearn import hmm
from nltk.sentiment import SentimentAnalyzer
from QuantConnect.Data.Custom.Tiingo import *
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import datetime
from datetime import timedelta
 
#region imports
from AlgorithmImports import *
#endregion
 

class HMMHybrid(QCAlgorithm):
 
    def Initialize(self):

        #timezone
        #self.SetTimeZone(TimeZones.EasternStandard)
        #Switch value for each regime
        self.switch = 'neutral'
        #self.SetWarmup(0)

        #have we just switched?
        self.curr = 'g'
        self.changed = True

        #sentiment analysis lookback days
        self.days = 4

        #self.SetSecurityInitializer(lambda x: x.SetMarketPrice(self.GetLastKnownPrice(x)))
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        #self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(1,00), self.xFineSelectionFunction)
        spy = self.AddEquity("SPY", Resolution.Hour)
        self.SetStartDate(2023,3,1)
        #self.SetEndDate(2023,4,1)
        self.SetCash(10_000_000)

        self.hmm_total = 10_000_000

        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.BeforeMarketClose("SPY"), self.MarketClose)
        #self.Schedule.On(self.DateRules.MonthStart("SPY"), \
        #         self.TimeRules.AfterMarketOpen("SPY"), \
        #         self.Reset)
        #self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(9,32), self.analyze_sentiment_for_fama_short)
 
        self.daily_return = 0
        self.prev_value = self.Portfolio.TotalPortfolioValue
        self.numberOfSymbols = 1000
 
        # Fama French Model
        self.symbols = [ spy.Symbol ]
        self.winsorize = 10
        self.num_fine = 50

        #rebalance
        # HMM stocks dictionary
        self.hmm_dict = {}
        self.recent_prices = {}

        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.AfterMarketOpen("SPY"), Action(self.rebalance))
        
        # Growth Multifactor Model
        self.numberOfSymbolsFine = 300
        self.num_portfolios = 6


 
    def CoarseSelectionFunction(self, coarse):
        CoarseWithFundamental = [x for x in coarse if x.HasFundamentalData and (float(x.Price)>10)]
 
        sortedByDollarVolume = sorted(CoarseWithFundamental, key=lambda x: x.DollarVolume, reverse=True)
        top = sortedByDollarVolume[:self.numberOfSymbols]
        return [i.Symbol for i in top]

    def FineSelectionFunction(self, fine):
        # FINE FILTERING FOR FRENCH STOCKS
 
        # drop stocks which don't have the information we need.
        # you can try replacing those factor with your own factors here
        filtered_fine = [x for x in fine if x.OperationRatios.OperationMargin.Value
                                        and x.ValuationRatios.PriceChange1M
                                        and x.ValuationRatios.BookValuePerShare]
       
 
        # rank stocks by three factor.
        sortedByfactor1 = sorted(filtered_fine, key=lambda x: x.ValuationRatios.PriceChange1M, reverse=True)
        sortedByfactor2 = sorted(filtered_fine, key=lambda x: x.ValuationRatios.BookValueYield, reverse=True)
        sortedByfactor3 = sorted(filtered_fine, key=lambda x: x.ValuationRatios.BookValuePerShare, reverse=False)
 
        stock_dict = {}
 
        # assign a score to each stock (ranking process)
        for i,ele in enumerate(sortedByfactor1):
            rank1 = i
            rank2 = sortedByfactor2.index(ele)
            rank3 = sortedByfactor3.index(ele)
            score = sum([rank1*0.2,rank2*0.4,rank3*0.4])
            stock_dict[ele] = score
 
        # sort the stocks by their scores
        self.sorted_stock = sorted(stock_dict.items(), key=lambda d:d[1],reverse=False)
        sorted_symbol = [x[0] for x in self.sorted_stock]
 
        # sort the top stocks into the long_list and the bottom ones into the short_list
        self.french_long = [x.Symbol for x in sorted_symbol[:self.num_fine]]
        self.french_short = [x.Symbol for x in sorted_symbol[-self.num_fine:]]
        for i in self.french_long:
            if i in self.french_short:
                self.french_long.remove(i)
                self.french_short.remove(i)

 
        #FINE FILTERING FOR GROWTH STOCKS
        filtered_fine = [x for x in fine if x.EarningReports.TotalDividendPerShare.ThreeMonths
                                        and x.ValuationRatios.PriceChange1M
                                        and x.ValuationRatios.BookValuePerShare
                                        and x.ValuationRatios.FCFYield]
 
        sortedByfactor1 = sorted(filtered_fine, key=lambda x: x.EarningReports.TotalDividendPerShare.ThreeMonths, reverse=True)
        sortedByfactor2 = sorted(filtered_fine, key=lambda x: x.ValuationRatios.PriceChange1M, reverse=False)
        sortedByfactor3 = sorted(filtered_fine, key=lambda x: x.ValuationRatios.BookValuePerShare, reverse=True)
        sortedByfactor4 = sorted(filtered_fine, key=lambda x: x.ValuationRatios.FCFYield, reverse=True)
 
        num_stocks = floor(len(filtered_fine)/self.num_portfolios)
 
        stock_dict = {}
 
        for i,ele in enumerate(sortedByfactor1):
            rank1 = i
            rank2 = sortedByfactor2.index(ele)
            rank3 = sortedByfactor3.index(ele)
            rank4 = sortedByfactor4.index(ele)
            score = [ceil(rank1/num_stocks),
                     ceil(rank2/num_stocks),
                     ceil(rank3/num_stocks),
                     ceil(rank4/num_stocks)]
            score = sum(score)
            stock_dict[ele] = score
        self.sorted_stock = sorted(stock_dict.items(), key=lambda d:d[1],reverse=True)
        sorted_symbol = [self.sorted_stock[i][0] for i in range(len(self.sorted_stock))]
        topFine = sorted_symbol[:self.num_fine]
        self.growth_long = [i.Symbol for i in topFine]

        for i in self.french_long:
            score = self.senti_stock(i)
            if score < 0.9:
                self.french_long.remove(i)

        for i in self.french_short:
            score = self.senti_stock(i)
            if score > 0.998 and score < 1.9:
                self.french_short.remove(i)
        
        for i in self.growth_long:
            score = self.senti_stock(i)
            if score < 0.9:
                self.growth_long.remove(i)

        return self.french_long + self.french_short + self.growth_long
        #end up with self.french_long, self.french_short, self.growth_long
 
    def analyze_sentiment(self, data, ticker):
        
        # Initialize sentiment analyzer
        sid = SentimentIntensityAnalyzer()
        #self.Debug("Data" + str(data))
 
        # Combine all news headlines and descriptions into a single string
        # Extract 'title' and 'description' columns from the DataFrame
        news_text = ' '.join([row['title'] + ' ' + row['description'] for _, row in data.iterrows()])
 
        # Convert to lowercase and split into words
        news_text = news_text.lower().split(" ")
 
        # Convert text to Unicode
        news_text = ' '.join(news_text)
        news_text = str(news_text.encode("utf-8"), "utf-8")
 
        # Perform sentiment analysis
        scores = sid.polarity_scores(news_text)
        #self.Debug(str(scores))
 
        # Extract compound sentiment score
        sentiment_score = scores['compound']
 
        #self.Debug(str(scores))
 
        #print(f"Sentiment analysis for {ticker}: {sentiment_score}")
 
        return sentiment_score
 
    def fetch_tiingo_news_data(self, ticker):
        self.tiingo_symbol = self.AddData(TiingoNews, ticker).Symbol
        #self.Debug(str(self.tiingo_symbol))

        # Historical data
        news_data = self.History(self.tiingo_symbol, self.days, Resolution.Daily)
        #self.Debug(f"We got {len(news_data)} items from our history request")
 
        return news_data
    
    #sentiment for each stock, called in individual factor model functions
    def senti_stock(self, ticker):

        data = self.fetch_tiingo_news_data(ticker)

        if ('title' not in data.iterrows()) or 'description' not in data.iterrows():
            return 2

        sentiment_score = self.analyze_sentiment(data, ticker)

        return sentiment_score
 
    def OnData(self, data):
        pass
 
    def rebalance(self): 
        #self.Debug('rebalancing line 257')
        #self.Debug(self.Securities.values())
        #self.Debug(len(self.Securities))
        next = self.next = self.train() # training hmm, output is in {bear, bull, neutral}
        #self.Debug('line 259: next = ' + next)
        if self.Portfolio.TotalHoldingsValue == 0: #if first time (ie, portfolio holdings is 0)
            self.Debug('line 261 portfolio empty')
            self.Debug(str(self.Portfolio.TotalHoldingsValue))

            self.hmm_dict = {}
            self.recent_prices = {}

            #self.changed = True ####needed?
            self.switch = next #assign switch var to be hmm output
            if self.switch == 'bear': #if bear, set curr to f, and run FF model
                self.curr = 'f'
                self.FamaFrench()
            else: #if bull or neutral, curr is g, run value model
                self.curr = 'g'
                self.GrowthModel()
            return
 
        self.switch = next #bear, neutral, or bull

        if self.curr == 'g': #if we ran growth yesterday...
            if self.switch == 'bear': #...but today its a bear mkt, then 
                self.Debug('growth to fama')
                self.changed = True #self.changed is true
                self.FamaFrench() #and run FF
                self.Debug('done with fama line 282')
                self.curr = 'f' #set curr to f
                return
            else: #...and still in a bull/neutral mkt
                self.Debug('still growth 286')
                self.changed = False #self.changed is false, and run growth model
                self.GrowthModel()
                self.Debug('done with growth line 289')
                return

        else: #if we ran FF yesterday (ie curr is f)
            if self.switch == 'bull': #and today is bull mkt, then changed is true, run growth, set curr to g
                self.Debug('fama to growth 295')
                self.changed = True
                self.GrowthModel()
                self.Debug('done w growth 298')
                self.curr = 'g'
                return
            else:#...and today is still bear/neutral, then changed is false, and run FF again
                self.Debug('still fama 302')
                self.changed = False
                self.FamaFrench()
                self.Debug('done with fama 305')
                return
 
    def FamaFrench(self):
        self.Debug("Fama French")
        self.Debug("port before fama " + str(self.Portfolio.TotalHoldingsValue))
        #total
        t = self.hmm_total
        #new list
        l1 = [x for x in self.french_long if self.Securities.ContainsKey(x)]
        l2 = [x for x in self.french_short if self.Securities.ContainsKey(x)]

        #new MO values
        q10 = t/len(l1) #positive value for new fama long
        q11 = -t/len(l2) #negative value for new fama short

        #if new regime, reset dict
        #else, reconfigure it all
        if self.changed == True:
            if len(self.hmm_dict.keys()) != 0:
                for i in self.hmm_dict.keys():
                    self.MarketOrder(i, -1 * self.hmm_dict[i] / self.recent_prices[i])
            self.Debug("new regime reset port: " + str(self.Portfolio.TotalHoldingsValue))
            self.hmm_dict = {}
            self.recent_prices = {}
            self.Debug("first fama run")
            
        else: #if still doing fama french
            #old MO values
            hmm_dic_vals = self.hmm_dict.values() #dict values as a list
            hdv = [*set(hmm_dic_vals)] #dict values no duplicates (should have one >0 entry and one <0 entry only)
            q = sorted(hdv) #sorted so q=[<0, >0]
            q00 = q[1] #positive value (from old fama long)
            q01 = q[0] #negative value (from old fama short)

            dic_keys = list(self.hmm_dict.keys())

            for i in dic_keys: #for i in dict !!!!!!!

                price = self.Portfolio[i].Price
                last_price = self.recent_prices[i]

                if ((i not in l1) and (i not in l2)) or (price == 0): #if not in french long or short

                    if self.hmm_dict[i] > 0: #if quantity > 0
                        self.MarketOrder(i, math.floor(-q00 / last_price)) #get rid of it
                        #self.Debug("get rid of stock (subtracting) " + str(math.floor(-q00 / last_price)))
                        #self.Debug(str(self.Portfolio.TotalHoldingsValue))
                    else: #if quantity < 0
                        self.MarketOrder(i, math.floor(-q01 / last_price)) #get rid of it
                        #self.Debug("get rid of stock (adding) " + str(math.floor(-q01 / last_price)))
                        #self.Debug(str(self.Portfolio.TotalHoldingsValue))
                    del self.hmm_dict[i]
                    del self.recent_prices[i]

                elif (i not in l1) and (i in l2): # if not in long but is in short

                    if self.hmm_dict[i] > 0: #if originally in long list and now shorting
                        s = math.floor((q11/price) - (q00/last_price))
                        if s != 0:
                            self.MarketOrder(i, s) #subtract long amount, add (negative) short amount
                            #self.Debug("alter stock (was long, now short): " + str(s))
                            #self.Debug(str(self.Portfolio.TotalHoldingsValue))
                        self.hmm_dict[i] = math.floor(q11)
                    else: #if originally in short list and still shorting
                        s = math.floor((q11/price) - (q01/last_price))
                        if s != 0:
                            self.MarketOrder(i, s) #get rid of old amount, add new (negativee) amount
                            #self.Debug("alter stock (was short, still shorting) " + str(s))
                            #self.Debug(str(self.Portfolio.TotalHoldingsValue))
                        self.hmm_dict[i] = math.floor(q11)

                    self.recent_prices[i] = price

                else: #if in long and not in short

                    if self.hmm_dict[i] > 0: #if was in long and still is long
                        s = math.floor((q10/price) - (q00/last_price))
                        if s != 0:
                            self.MarketOrder(i, s) #subtract old long amount, add new long amount
                            #self.Debug("alter stock (was long, still long): " + str(s))
                            #self.Debug(str(self.Portfolio.TotalHoldingsValue))
                        self.hmm_dict[i] = math.floor(q10)
                    elif self.hmm_dict[i] < 0: #if was in short and now is long
                        s = math.floor((q10/price) - (q01/last_price))
                        if s != 0:
                            self.MarketOrder(i, s) #subtract old (negative) short amount, add new long amount
                            #self.Debug("alter stock (was short, now long): " + str(s))
                            #self.Debug(str(self.Portfolio.TotalHoldingsValue))
                        self.hmm_dict[i] = math.floor(q10)
                
                
                    #after changing market orders and hmm_dict, put prices in recent_prices dict
                    
                    self.recent_prices[i] = price

        for i in l1: #for i in long list
            price = self.Portfolio[i].Price
            if price == 0:
                continue
            elif i not in list(self.hmm_dict.keys()): #if not yet covered (not in dict)
                self.MarketOrder(i, math.floor(q10 / price)) #add new long amount
                #self.Debug("new long: " + str(math.floor(q10 / price)))
                #self.Debug(str(self.Portfolio.TotalHoldingsValue))
                self.hmm_dict[i] = math.floor(q10)
                #add price to recent_prices
                self.recent_prices[i] = price

        for i in l2: #for i in short list
            price = self.Portfolio[i].Price
            if price == 0:
                continue
            elif i not in list(self.hmm_dict.keys()): #if not yet covered (not in dict)
                self.MarketOrder(i, math.floor(q11 / price)) #add new (negative) short amount
                #self.Debug("new short: " + str(math.floor(q11 / price)))
                #self.Debug(str(self.Portfolio.TotalHoldingsValue))
                self.hmm_dict[i] = math.floor(q11)
                #add price to recent_prices
                self.recent_prices[i] = price


        self.Debug("final holdings after fama run: " + str(self.Portfolio.TotalHoldingsValue))
        #self.Debug('hmm dict: ' + str(len(self.hmm_dict.values())) + ' ' + str(self.hmm_dict.values()))

 
    def GrowthModel(self):
        self.Log("Growth Model")
        self.Debug("before growth " + str(self.Portfolio.TotalHoldingsValue))

        #total
        t = self.hmm_total #t is total allocation
        #new MO values
        q1 = t/len(self.growth_long) #total allocation / number of stocks
        #we will equally distribute our allocation amongst each stock we are longing
        #new list
        l1 = [x for x in self.growth_long if self.Securities.ContainsKey(x)]

        # if just changed regimes, reset dict and add from scratch
        # else recalculate dict from old dict
        if self.changed == False:

            q0 = t/len(self.hmm_dict)
            #self.Debug("old value to change " + str(q0))

            #self.Log("old dict " + str(self.hmm_dict.keys()))
            #self.Log(str(self.hmm_dict.values()))
            dic_keys = list(self.hmm_dict.keys())

            for i in dic_keys: #for i in dict
                price = self.Portfolio[i].Price
                last_price = self.recent_prices[i]

                if (i not in l1) or (price == 0): #if not in growth_long or price 0 (ie, we no longer want it in hmm strat)
                    self.MarketOrder(i, math.floor(-q0 / last_price)) #get rid of it
                    del self.hmm_dict[i]
                    del self.recent_prices[i]
                else: #if in growth long (ie, still want it)
                    s = math.floor((q1/price) - (q0/last_price))
                    if s != 0:
                        self.MarketOrder(i, s) #adjust quantity accordingly
                    self.hmm_dict[i] = math.floor(q1)

                    #after changing market orders and hmm_dict, put prices in recent_prices dict
                    self.recent_prices[i] = price
        
        else: #if self.chaged is true
            if len(self.hmm_dict.keys()) != 0:
                for i in self.hmm_dict.keys():
                    self.MarketOrder(i, -1 * self.hmm_dict[i] / self.recent_prices[i])
            self.Debug("new regime reset port: " + str(self.Portfolio.TotalHoldingsValue))#reset portfolio
            self.hmm_dict = {} #reset hmm_dict and recent prices
            self.recent_prices = {}

        for i in l1: #for i in growth_long
            price = self.Portfolio[i].Price
            if price == 0:
                continue
            elif i not in list(self.hmm_dict.keys()): #if not in dict (ie, if new stock we want to include)
                self.MarketOrder(i, math.floor(q1 / price)) #add it in
                self.hmm_dict[i] = math.floor(q1)
                #add price to recent_prices
                self.recent_prices[i] = price
        
        #self.Log("new dict " + str(self.hmm_dict.keys()))
        #self.Log(str(list(self.hmm_dict.values())))
        self.Debug("after growth: " + str(self.Portfolio.TotalHoldingsValue))

 
    def MarketClose(self):
        self.daily_return = 100*((self.Portfolio.TotalPortfolioValue - self.prev_value)/self.prev_value)
        self.prev_value = self.Portfolio.TotalPortfolioValue
        self.Log(self.daily_return)
        self.Log("Switch: {}".format(self.switch))
        return
 
    def train(self):
       # Hidden Markov Model Modifiable Parameters
        hidden_states = 3;
        em_iterations = 75;
        data_length = 3356;
        # num_models = 7;
 
        #history = self.History("SPY", 2718, Resolution.Daily)
        #prices = list(history.loc["SPY"]['close'])
 
        history = self.History(self.symbols, 2600, Resolution.Daily)
        for symbol in self.symbols:
            if not history.empty:
                # get historical open price
                prices = list(history.loc[symbol.Value]['close'])
 
        # Volatility is computed by obtaining variance between current close and
        # prices of past 10 days
        Volatility = []
 
        # MA is the 10 day SMA
        MA = []
 
        # Return is the single-day percentage return
        Return = []
        ma_sum = 0;
 
        # Warming up data for moving average and volatility calculations
        for i in range (0, 10):
            Volatility.append(0);
            MA.append(0);
            Return.append(0);
            ma_sum += prices[i];
        # Filling in data for return, moving average, and volatility
        for i in range(0, len(prices)):
            if i >= 10:
                tail_close = prices[i-10];
                prev_close = prices[i-1];
                head_close = prices[i];
                ma_sum = (ma_sum - tail_close + head_close);
                ma_curr = ma_sum/10;
                MA.append(ma_curr);
                Return.append(((head_close-prev_close)/prev_close)*100);
                #Computing Volatility
                vol_sum = 0;
                for j in range (0, 10):
                    curr_vol = abs(ma_curr - prices[i-j]);
                    vol_sum += (curr_vol ** 2);
                Volatility.append(vol_sum/10);
 
        prices = prices[10:]
        Volatility = Volatility[10:]
        Return = Return[10:]
 
        # Creating the Hidden Markov Model
        model = hmm.GaussianHMM(n_components = hidden_states,
                                covariance_type="full", n_iter = em_iterations);
 
        obs = [];
        for i in range(0, len(Volatility)):
            arr = [];
            arr.append(Volatility[i]);
            arr.append(Return[i]);
            obs.append(arr);
 
        # Fitting the model and obtaining predictions
        model.fit(obs)
        predictions = model.predict(obs)
 
        # Regime Classification
        regime_vol = {};
        regime_ret = {};
 
        for i in range(0, hidden_states):
            regime_vol[i] = [];
            regime_ret[i] = [];
 
        for i in range(0, len(predictions)):
            regime_vol[predictions[i]].append(Volatility[i]);
            regime_ret[predictions[i]].append(Return[i]);
 
        vols = []
        rets = []
        today_regime = predictions[-1]
        for i in range(0, hidden_states):
            vol_dist = Distribution()
            vol_dist.Fit(regime_vol[i])
            vols.append(vol_dist.PDF(Volatility[-1]))
            ret_dist = Distribution()
            ret_dist.Fit(regime_ret[i])
            rets.append(ret_dist.PDF(Return[-1]))
 
        # > 0.5 Low-Pass Filter
        bear = -1
        bull = -1
        neg_return = 1
        pos_return = -1
        low_vol = 100
        for i in range(0, hidden_states):
            if sum(regime_ret[i]) / len(regime_ret[i]) < neg_return:
                neg_return = sum(regime_ret[i]) / len(regime_ret[i])
                bear = i
            if sum(regime_ret[i]) / len(regime_ret[i]) > pos_return:
                pos_return = sum(regime_ret[i]) / len(regime_ret[i])
                bull = i
 
        if vols[today_regime] / sum(vols) >= 0.3 and rets[today_regime] / sum(rets) >= 0.5:
            if bear == today_regime:
                return 'bear'
            else:
                return 'bull'
        else:
            return 'neutral'
 
# Kolmogorov-Smirnov Test to find best distribution
class Distribution(object):
 
    def __init__(self, dist_names_list = []):
        self.dist_names = ['norm','lognorm','expon', 'gamma',
                           'beta', 'rayleigh', 'norm', 'pareto']
        self.dist_results = []
        self.params = {}
 
        self.DistributionName = ""
        self.PValue = 0
        self.Param = None
        self.isFitted = False
 

    def Fit(self, y):
        self.dist_results = []
        self.params = {}
        for dist_name in self.dist_names:
            dist = getattr(scipy.stats, dist_name)
            param = dist.fit(y)
            self.params[dist_name] = param
            #Applying the Kolmogorov-Smirnov test
            D, p = scipy.stats.kstest(y, dist_name, args=param);
            self.dist_results.append((dist_name,p))
 
        #select the best fitted distribution
        sel_dist,p = (max(self.dist_results,key=lambda item:item[1]))
        #store the name of the best fit and its p value
        self.DistributionName = sel_dist
        self.PValue = p
        self.isFitted = True
 
        return self.DistributionName, self.PValue
 
    def PDF(self, x):
        dist = getattr(scipy.stats, self.DistributionName)
        n = dist.pdf(x, *self.params[self.DistributionName])
        return n