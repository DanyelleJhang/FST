import numpy as np
import pandas as pd

def get_time_step(data:pd.DataFrame):
    time_diffs = data.index.to_series().diff().dropna()
    time_units = {
        'S': pd.Timedelta(seconds=1),
        'T': pd.Timedelta(minutes=1),
        'H': pd.Timedelta(hours=1),
        'D': pd.Timedelta(days=1),
        'W': pd.Timedelta(weeks=1),
        'MS': pd.Timedelta(days=30),
        'YS': pd.Timedelta(days=365),
    }

    def get_min_i_for_unit(unit_name):
        unit = time_units[unit_name]
        min_i = time_diffs.apply(lambda x: x / unit).mean() # min 修正成 mean
        return min_i

    results = {unit: get_min_i_for_unit(unit) for unit in time_units.keys()}


    filtered_values = {k: v for k, v in results.items() if v > 1} # 找出大於1的時間單位

    time_interval = min(filtered_values.values()) # 求出時間間隔
    time_interval = abs(time_interval) # 這邊取整數，但BUG也可能在這 # 目前不知道要四捨五入還是無條件進位或無條件捨取，不過目前測試四捨五入的狀況比較好
    time_unit = min(filtered_values, key=filtered_values.get) # 找出最小時間單位

    print("最小時間單位",time_unit,"最小時間間隔",time_interval)
    return time_interval, time_unit


def frequency_preprocessing(data:pd.DataFrame, time_interval:str, time_unit:str,operator:list,NanImputation:bool):
    # YS(年初), MS(月初), W(周), D(日), H(小時), T(分鐘), S(秒),
    data = data.select_dtypes(include=['int', 'float']) # 保留只有數字的欄位
    aggregation_functions = {}
    for i in data.columns:
        aggregation_functions[i]=operator
    data = data.resample(f'{time_interval}{time_unit}').agg(aggregation_functions)
    data.columns = [f"{col[0]}({col[1]})" for col in data.columns.values]
    #print(data)
    if NanImputation:
        data.interpolate(method='linear', inplace=True)
        return data
    else:
        return data



"""程式範例
df = pd.read_csv("C:/Users/foresight_User/Data/測試資料/VISERA_170.csv")
# 時間欄位
datetime_col = "Metrology Start Time"

# 將 datetime_col 轉換為日期時間格式，並設置為索引
df[datetime_col] = pd.to_datetime(df[datetime_col], format='%Y/%m/%d %H:%M')
df = df.set_index(datetime_col)


# 獲取建議的時間間隔和時間單位(不可低於以下時間單位和間隔)
suggest_time_interval , suggest_time_unit= get_time_step(data=df)


# 定義聚合操作列表 (可選)
operator_list = ['first', 'last','mean','prod','median','max','min','sum'] 

# 頻率預處理

df_freq = frequency_preprocessing(data=df, time_interval=suggest_time_interval, time_unit=suggest_time_unit,operator=operator_list,NanImputation=True)

"""