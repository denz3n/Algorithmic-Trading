#region imports
from AlgorithmImports import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
from numpy import unravel_index
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing

from statsmodels.tsa.stattools import coint, adfuller

import statsmodels.api as sm

from scipy import stats

#endregion
class ClusterSelection(QCAlgorithm):
    
    filteredByPrice = None
    
    def Initialize(self):
        self.SetStartDate(2017, 1, 1)  
        self.SetEndDate(2017, 1, 8) 
        #self.SetCash(100000)
        
        self.AddUniverse(self.CoarseSelectionFilter, self.FineSelectionFunction)
        """
        # rebalance our portfolio at the end of every week 
        self.Schedule.On(self.DateRules.WeekEnd("SPY"),
                 self.TimeRules.BeforeMarketClose("SPY", 30),
                 self.Rebalance)
        """
        # run the clustering thing every week 
        self.Schedule.On(self.DateRules.WeekStart(), 
            self.TimeRules.At(0, 0), 
            Action(self.GetCluster))

        self.UniverseSettings.Resolution = Resolution.Daily
        #### the parameters specific to clustering pairs trading #####
        self.numberOfSymbols = 1000
        self.min_sample = 4
        ##############################################################
        self.Debug("Start time " + str(type(self.StartDate)))

        
      

    
    def CoarseSelectionFilter(self, coarse):
        # select the stocks that has fundamental data + more than 5 dollar
        CoarseWithFundamental = [x for x in coarse if x.HasFundamentalData and (float(x.Price)>5)]
        sortedByDollarVolume = sorted(CoarseWithFundamental, key=lambda x: x.DollarVolume, reverse=True)
        top = sortedByDollarVolume[:self.numberOfSymbols]
        return [i.Symbol for i in top]  

    
        
    def FineSelectionFunction(self, fine):
        #remove any stock with less than 250 market cap
        MarketCapFilter = [x for x in fine if float(x.MarketCap)> 250_000_000]
        sortedByMarketCap = sorted(MarketCapFilter, key=lambda c: c.MarketCap, reverse=True)
        filteredFine = [i.Symbol for i in sortedByMarketCap]

        self.filter_fine = filteredFine[:self.numberOfSymbols]
        return self.filter_fine
    

    def OnData(self, data):
         
        #self.Log({'","'.join([key.Value for key in data.Keys])})

        pass


    
    def GetCluster(self):
        
        self.Debug(self.Time)
        qb = self
        # looking back for 1 year  
        history = qb.History(self.filter_fine, 365, Resolution.Daily)
        marketcap = []
        sector_code = []
        

        if history.empty: 
            return
        
        df = history['close'].unstack(level =0)
        dg=df.apply(lambda x: x.pct_change(), axis=0)
        #self.Debug('allcolumns:{}'.format(dg.shape))
        dg = dg.dropna()
        #self.Debug('dropcolumns:{}'.format(dg.shape))
        N_PRIN_COMPONENTS = 10

        ### get the fundamentals based on the tickers we're taking 
        for i in dg.columns:
            marketcap.append(self.Securities[i].Fundamentals.MarketCap)
            sector_code.append(self.Securities[i].Fundamentals.AssetClassification.MorningstarSectorCode)

        
        
        ### if we have more components we want, then we do PCA, else don't do 
        if dg.shape[0] > N_PRIN_COMPONENTS:
            pca = PCA(n_components=N_PRIN_COMPONENTS)
            pca.fit(dg)
            self.Log(pca.components_.T.shape)
            self.Log(np.array(marketcap).reshape(len(marketcap), 1).shape)
        
            X = np.hstack(
                (pca.components_.T,
                np.array(marketcap).reshape(len(marketcap), 1), 
                np.array(sector_code).reshape(len(sector_code), 1)
                ))
        else: 
            X = np.hstack(
                (dg.T,
                np.array(marketcap).reshape(len(marketcap), 1), 
                np.array(sector_code).reshape(len(sector_code), 1)
                ))
        
        ### divide by each sector match stock ticker to sector 
        sector_dict = dict.fromkeys(set(sector_code), None)
        for uni_sector in set(sector_code): 
            #print(uni_sector)
            ind = [i for i, n in enumerate(sector_code) if n == uni_sector]
            #print(len(ind))
            v =[dg.columns[i] for i in ind]
            #print(len(v))
            sector_dict[uni_sector] = v
        
        ### cluster within each sector
        sector_cluster_assignment = {}
        for sector in set(sector_code):
            #print(sector)
            a =  X[X[:, -1] == sector]
            a = preprocessing.StandardScaler().fit_transform(a)
            clustering = OPTICS(min_samples=self.min_sample).fit(a)
            labels = clustering.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            clustered_series_all = pd.Series(index=sector_dict[sector], data=labels.flatten())
            #clustered_series = pd.Series(index=dg.columns, data=labels.flatten())
            sector_cluster_assignment[sector] = clustered_series_all
            #print(n_clusters_)
            self.Debug('sector:'+ str(sector) + 'cluster_number :' + str(n_clusters_))
        
        ## for each sector, each cluster, pick the one pair that meets both criteria 
        sector_cluster_matrix = dict.fromkeys(set(sector_code), None)
        for sector,cluster_res in sector_cluster_assignment.items():
            print(sector)
            #print(cluster_res.value_counts())
            cluster_dict = {}
            clustered_series = cluster_res[cluster_res != -1]
            #print(clustered_series)
            for i, which_clust in enumerate(clustered_series.value_counts().index):
                tickers = clustered_series[clustered_series == which_clust].index
                #print(tickers)
                score_matrix, pvalue_matrix, pairs, adf_matrix, adf_p_matrix, adf_pair = self.find_cointegrated_pairs(
                    df[tickers]
                )
                cluster_dict[which_clust] = {}
                sat_pair = list(set(pairs).intersection(adf_pair)) # pairs that satisifes both conditions 
                if len(sat_pair)>1:
                    cluster_dict[which_clust]['final_pair'] = self.compare_pairs(sat_pair, tickers, pvalue_matrix,  adf_p_matrix)
                elif len(sat_pair) == 1:
                    cluster_dict[which_clust]['final_pair'] = sat_pair[0]
                else: 
                    cluster_dict[which_clust]['final_pair'] = None
                
            sector_cluster_matrix[sector] = cluster_dict
        
        total_pairs = []
        for sector, sector_info in sector_cluster_matrix.items():
            for cluster, matrix in sector_info.items():
                self.Debug(matrix['final_pair'])
                if matrix['final_pair'] is not None: 
                    total_pairs.append(matrix['final_pair'] )
                
    def find_cointegrated_pairs(self, data, significance=0.05):
        n = data.shape[1]
        score_matrix = np.zeros((n, n))
        pvalue_matrix, adf_matrix, adf_p_matrix = np.ones((n, n))
        keys = data.keys()
        #print(keys)
        pairs, adf_pairs= []
        for i in range(n):
            for j in range(i+1, n):
                S1 = data[keys[i]]
                S2 = data[keys[j]]
                result = coint(S1, S2)
                score = result[0]
                pvalue = result[1]
                score_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
                model = sm.OLS(S1,S2)
                results = model.fit()
                res = results.resid
                adf = adfuller(res)
                adf_matrix[i, j] = adf[0]
                adf_p_matrix[i, j] = adf[1]
                # if the cointegration test is significant, append the pair
                if pvalue < significance:
                    pairs.append((keys[i], keys[j]))
                # if the adf test is significant, append the pair
                if adf[1] < significance:
                    adf_pairs.append((keys[i], keys[j]))
        return score_matrix, pvalue_matrix, pairs, adf_matrix, adf_p_matrix, adf_pairs

        
    def compare_pairs(self, pair_list, tick_order, p_val, adf_p_val):
        """
        If we have more than 1 pair that satisfy the condition, we compare them 
        """
        p_val_total = []
        for i in pair_list: 
            #print(i)
            ticker_ord = tick_order.tolist()
            row = ticker_ord.index(i[0])
            col = ticker_ord.index(i[1])
            #print(row, col)
            coint_p = p_val[row, col]
            adf_p =  adf_p_val[row, col]
            p_val_total.append(coint_p+adf_p)
        min_ind = np.array(p_val_total).argmin()
        return pair_list[min_ind]

    
            

    def TradePairs(self): 
        pass
