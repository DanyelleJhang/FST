import pandas as pd
from typing import Union
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.api import VAR

import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

class VectorAutoregression:
    def __init__(self):
        self.data = None
        self.results = None
        self.summary = None
    def remove_collinearity(self,data,remain:list=None)-> pd.DataFrame: #condition
        """
        移除資料中的共線性特徵

        參數:
        data : pd.DataFrame
            原始資料。
        remain : list, 可選
            要保留的特徵。

        返回:
        pd.DataFrame
            移除共線性特徵後的資料。
        """
        # remain 保留Y值
        if data.shape[1] <2:
            return {"info": ["The dataset must have at least 2 dimensions."]}
        
        if (remain != None) and data.shape[1] > 2 :
            remained_data = data.loc[:,remain]

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.ffill().bfill() 
        vif_data = pd.DataFrame()
        vif_data["feature"] = data.columns
        vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]


        # 去除VIF為NAN或INF
        vif_data = vif_data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        data = data[vif_data["feature"].to_list()]


        # 保留低相關的 feature
        from itertools import combinations
        relevance_threshold = 0.3
        coef_dataframe = data.corr()
        remain_col_index = ()
        for i_1,i_2 in list(combinations(coef_dataframe,2)):
            coef = coef_dataframe.loc[i_1,i_2]
            if abs(coef) < abs(relevance_threshold):
                remain_col_index += (i_1,i_2)
        feature_index = list(set(remain_col_index))  

        
        data = data[feature_index]
        if remain != None:
            remain_list = list(set(remain) - set(feature_index))
            if (data.shape[1] > 2) and (len(remain_list)>0):
                remained_data = remained_data.loc[:,remain_list]
                data = pd.concat([data,remained_data],axis=1)
        #data = data.fillna(method='ffill').fillna(method='bfill')
        if data.shape[1] >= 2:
            return data
        else:
            return {"info": ["Data exhibits multicollinearity, making analysis impossible"]}


    def cal_maxLag(self,data:pd.DataFrame):
        
        n_totobs = len(data)
        ntrend = 1 #len(trend) if trend.startswith("c") else 0
        neqs = data.shape[1]
        max_estimable = (n_totobs - neqs - ntrend) // (1 + neqs)
        if max_estimable > 1:
            return max_estimable
        else:
            return 1
    def fit(self,data:pd.DataFrame,maxlags:Union[int,str]="auto",ic:str=None,remain:list=None):
        data = data.copy()
        clean_data = self.remove_collinearity(data,remain)
        #print(data)
        #print(not data.empty)
        if not isinstance(clean_data,dict):
            model = VAR(clean_data)
            # ==== 這邊不要動 =====

            """ 
            這是套件設定的
            trend : str {"n", "c", "ct", "ctt"}
                * "n" - no deterministic terms
                * "c" - constant term
                * "ct" - constant and linear term
                * "ctt" - constant, linear, and quadratic term

            maxlags 不可以超過 max_estimable 的值
            maxlags 為模型擬合最大數值
            statemodel有設定條件，已經寫在下述的程式
            使用者要調整低於 maxlags
            """
            max_estimable = self.cal_maxLag(clean_data)
            # ==== 這邊不要動 =====

            print("maxlags 要小於等於: ", max_estimable)
            if maxlags == "auto":
                maxlags = max_estimable
            if maxlags > max_estimable:
                #raise Exception(" maxlags 要小於等於: ", max_estimable)
                return {"info": [f"MaxLags must be less than or equal to: {max_estimable}"]}


            """
            ic 為評估模型的好壞
            ic = {'aic', 'fpe', 'hqic', 'bic', None}
            Information criterion to use for VAR order selection.
            aic : Akaike
            fpe : Final prediction error
            hqic : Hannan-Quinn
            bic : Bayesian a.k.a. Schwarz
            """

            self.results = model.fit(maxlags=maxlags, ic=ic)
            self.data = clean_data
            print(f"在最大 lag 數目為 {max_estimable} 的情況下，VAR 找出的最佳 lag 為: ",self.results.k_ar)
        else:
            return clean_data
        
    def getRelationship(self,target:str,pvalue_threshold:float=0.05)-> pd.DataFrame:

        """
        獲取指定目標變量與其他特徵之間的關係。

        參數:
        target : str
            客戶關心的目標變量(Y)。
        pvalue_threshold : float, 可選
            顯著性水平的閾值，默認為0.05。

        返回:
        pd.DataFrame
            包含目標變量與其他特徵之間的係數和p值的數據框。
        """
        if isinstance(self.results,VARResultsWrapper):
            coef_df = self.results.params[target]
            pvalues_df = self.results.pvalues[target]

            # 合并系数和p值
            summary = pd.concat([coef_df, pvalues_df], axis=1)
            summary.columns = ['coef', 'pvalue']
            summary = summary.drop(index="const").reset_index()
            summary_index = summary["index"].str.split(".", expand=True).rename(columns={0:"time_lag",1:"feature"})
            if summary.empty:
                return {"info": ["there is no results found from VAR"]}
            else:
                summary_index["time_lag"] = summary_index["time_lag"].str.replace("L","").astype(int)
                summary = pd.concat([summary_index,summary],axis=1).drop(columns="index")
                summary = summary[summary["pvalue"]<pvalue_threshold].reset_index(drop=True)
                self.summary = summary
                return summary
        else:
            return {"info": ["there is no significant time lag"]}
    def shift_transform(self,data,remain_origin:list=None,exclude_lag:list=None,reference:pd.DataFrame=None)-> pd.DataFrame:
        """
        remain_origin
        : the output dataframe will contains original feature
            > remain_origin = None
                X_1_lag_1, X_1_lag_2, ....

            > remain_origin = X_1
                X_1, X_1_lag_1, X_1_lag_2, ....

        exclude_lag
        : the output dataframe will exclude feature with lagged time
            > exclude_lag = None
                X_1_lag_1, X_1_lag_2, X_2_lag_1, X_2_lag_2....
            > exclude_lag = X_1
                X_1, X_2_lag_1, X_2_lag_2....
        """
        summary = self.summary
        data = data.copy()
   
        if summary is None:
            return {"info": ["due to there is no significant time lag, data could not be transformed"]}

        if exclude_lag:
            summary = summary[~summary["feature"].isin(exclude_lag)]
        
        def apply_lag(row,reference):
            feature, time_lag = row["feature"], row["time_lag"]
            feature_lag_name = f"{feature}_lag_{time_lag}"
            data[feature_lag_name] = data[feature].shift(time_lag)

            if reference is not None:
                fill_values = reference[feature][-time_lag:].to_list()
                data.loc[data[feature_lag_name].isna(), feature_lag_name] = fill_values

        summary.apply(lambda row: apply_lag(row, reference), axis=1)
        
        if reference is None:
            data.dropna(inplace=True)
            

        columns_list = summary["feature"].tolist()

        if remain_origin:
            columns_list = list(set(columns_list) - set(remain_origin))
        
        data_transform = data.drop(columns=columns_list)
        return data_transform.reset_index()



"""範例

data = pd.read_csv(....)
data = data.set_index("TimeTag")


maxlags = ....
target= ....

# 假設後十筆維test資料
train_data = data.iloc[:-10,:]
test_data = data.iloc[-10:,:]

var = VectorAutoregression()

var.fit(data=train_data,maxlags=maxlags,ic=None)

var_result = var.getRelationship(target=target)

# train_data 轉換
var.shift_transform(data=train_data,remain_origin=None,exclude_lag=[target],reference=None)

# test_data 轉換，train_data進行補植
var.shift_transform(data=test_data,remain_origin=None,exclude_lag=[target],reference=train_data)

"""