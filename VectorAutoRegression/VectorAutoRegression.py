# some example data
import pandas as pd
from typing import Union
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.api import VAR

from statsmodels.tsa.base.datetools import dates_from_str



def time_series_format_preprocessing(df:pd.DataFrame, datetime_col:str):
    #
    # YS(年初), MS(月初), W(周), D(日), H(小時), T(分鐘), S(秒),
    #
    # df = df.copy()
    # df[datetime_col] = pd.to_datetime(df[datetime_col], format='%Y-%m-%d %H:%M:%S')
    # df = df.set_index(datetime_col)
    # df = df.asfreq(interval)
    # return df if set_index_flag else df.reset_index(drop=False)
    df.index =pd.DatetimeIndex(dates_from_str(data[datetime_col]))
    df = df.drop(columns=[datetime_col])
    """
    Imputes missing values in a time series DataFrame using the specified function.

    Args:
        df (pd.DataFrame): The input DataFrame containing the time series data.
        func (str): The imputation function to be applied. Supported options are:
                    - 'forward_fill': Forward fill missing values.
                    - 'backward_fill': Backward fill missing values.
                    - 'moving_average': Impute missing values using moving average.
                    - 'interpolation': Perform linear interpolation to fill missing values.
        col_name (str): The name of the column to impute missing values.

    Returns:
        pd.DataFrame: The DataFrame with missing values imputed based on the specified function.
    """
    return df


def time_series_impute_missing_value(df, func, col_name):
    """
    Imputes missing values in a time series DataFrame using the specified function.

    Args:
        df (pd.DataFrame): The input DataFrame containing the time series data.
        func (str): The imputation function to be applied. Supported options are:
                    - 'forward_fill': Forward fill missing values.
                    - 'backward_fill': Backward fill missing values.
                    - 'moving_average': Impute missing values using moving average.
                    - 'interpolation': Perform linear interpolation to fill missing values.
        col_name (str): The name of the column to impute missing values.

    Returns:
        pd.DataFrame: The DataFrame with missing values imputed based on the specified function.
    """
    df = df.copy()
    if func == 'forward_fill':
        df[col_name].ffill(inplace=True)
    elif func == 'backward_fill':
        df[col_name].bfill(inplace=True)
    elif func == 'moving_average':
        df[col_name].fillna(df[col_name].rolling(window=3, min_periods=1).mean(), inplace=True)
    elif func == 'interpolation':
        df[col_name].interpolate(inplace=True)
    else:
        raise ValueError(f"Unsupported imputation function: {func}")

    return df

def cal_maxLag(data:pd.DataFrame):
    n_totobs = len(data)
    ntrend = 1 #len(trend) if trend.startswith("c") else 0
    neqs = data.shape[1]
    max_estimable = (n_totobs - neqs - ntrend) // (1 + neqs)
    return max_estimable


def vectorAutoregression(data:pd.DataFrame,maxlags:Union[int,str]="auto",ic:str=None):
    model = VAR(data)
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
    max_estimable = cal_maxLag(data)
    # ==== 這邊不要動 =====

    print(" maxlags 要小於等於: ", max_estimable)

    if maxlags == "auto":
        maxlags = max_estimable
    if maxlags > max_estimable:
        raise Exception(" maxlags 要小於等於: ", max_estimable)


    """
    ic 為評估模型的好壞
    ic = {'aic', 'fpe', 'hqic', 'bic', None}
    Information criterion to use for VAR order selection.
    aic : Akaike
    fpe : Final prediction error
    hqic : Hannan-Quinn
    bic : Bayesian a.k.a. Schwarz
    """

    results = model.fit(maxlags=maxlags, ic=ic)
    print(f"在最大 lag 數目為 {max_estimable} 的情況下，VAR 找出的最佳 lag 為: ",results.k_ar)
    return results


def vectorAutoregressionRelationship(results:VARResultsWrapper,target:str,pvalue_threshold:float=0.05):
    # target  客人關心的 Y 是甚麼，Y 會包含在 results 中

    coef_df = results.params[target]
    pvalues_df = results.pvalues[target]

    # 合并系数和p值
    summary = pd.concat([coef_df, pvalues_df], axis=1)
    summary.columns = ['coef', 'pvalue']
    summary = summary.drop(index="const").reset_index()
    summary_index = summary["index"].str.split(".", expand=True).rename(columns={0:"time_lag",1:"feature"})
    summary_index["time_lag"] = summary_index["time_lag"].str.replace("L","").astype(int)
    summary = pd.concat([summary_index,summary],axis=1).drop(columns="index")
    summary = summary[summary["pvalue"]<pvalue_threshold].reset_index(drop=True)
    return summary





"""
如果資料是每日資料（daily data），lag 1 表示一天。
如果資料是月度資料（monthly data），lag 1 表示一個月。
如果資料是年度資料（annual data），lag 1 表示一年。
"""
"""
讓使用者決定好奇的target是甚麼，套件會自動找出所有跟target相關的不同time lag的時間變相
maxlags 會影響最終最佳推薦的結果，也得慎選，但設定上不可以超過 cal_maxLag 輸出的值
"""

"""
讀取資料
data = pd.read_csv(dir+"National Stock Exchange/tcs_stock.csv").drop(columns=["Symbol","Series"])
data = time_series_format_preprocessing(data,"Date")
data = time_series_impute_missing_value(data, 'interpolation', "Volume")
"""

"""
maxlags = cal_maxLag(data)
-> user 不能指定超過這個的LAG，LAG值也要大於0
-> user 也可以自訂maxlag
results = vectorAutoregression(data,maxlags=maxlags,ic="fpe")

results.summary()
-> 顯示所有時間差的摘要

target = "Trades"
-> 讓使用者選則想知道的Y 
pvalue_threshold = 0.05

VAR_relationship = vectorAutoregressionRelationship(results=results,target=target,pvalue_threshold=pvalue_threshold)
"""

"""
formula_parts = []
for index, row in VAR_relationship.iterrows():
    coef = row['coef']
    if coef < 0:
        term = f"- {-coef:.3f} * {row['feature']}(t-{row['time_lag']})"
    else:
        term = f"{coef:.3f} * {row['feature']}(t-{row['time_lag']})"
    formula_parts.append(term)

formula = " + ".join(formula_parts).replace("+ -", "- ")
target_formula = f"f{target}(t) = {formula}"

# 打印公式


Y1(t) = coef*Y1(t-best_lag_1) + coef*Y1(t-best_lag_2) + coef*Y2(t-best_lag_3) + ....

print(target_formula)
fTrades(t) = 0.553 * Trades(t-1)


"""