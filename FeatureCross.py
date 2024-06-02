import torch # '2.3.0+cpu'
from torch import nn # 
import numpy as np # '1.26.4'
import pandas as pd # '1.3.4'
import warnings
import random
warnings.filterwarnings('ignore')
#from itertools import combinations
from collections import defaultdict
from joblib import Parallel, delayed # '1.3.2'
from tqdm import tqdm # '4.64.1'
from typing import Union
from sklearn.model_selection import train_test_split
n_jobs = -1

""" version
torch == '2.3.0+cpu'
numpy == '1.26.4'
pandas == '1.3.4'
joblib == '1.3.2'
tqdm == '4.64.1'
scipy == '1.11.4'
"""


class Regression(nn.Module):
    def __init__(self, alg_type,input_dim, weights =None, bias = None):
        super(Regression, self).__init__()
        # 定義層
        self.alg_type = alg_type
        if self.alg_type =="LogisticRegression":
            self.linear = nn.Linear(input_dim, 2)
        if self.alg_type =="LinearRegression":
            self.linear = nn.Linear(input_dim, 1)
        if (weights != None) and (bias != None):
            # 使用原始权重和偏差初始化
            with torch.no_grad():
                self.linear.weight = torch.nn.Parameter(weights)
                self.linear.bias = torch.nn.Parameter(bias)
    def forward(self, x):
        # 預測值
        if self.alg_type =="LogisticRegression":
            y_predicted = torch.sigmoid(self.linear(x))
        if self.alg_type =="LinearRegression":
            y_predicted = self.linear(x)
        return y_predicted

# 建立模型
def train(alg,alg_type,criterion, X_train,y_train,num_epochs,learning_rate):
    model = alg(alg_type,input_dim = X_train.shape[1])

    # 定義損失函數
    # criterion = nn.BCELoss()

    # 定義優化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 訓練模型
    total_loss = 0 
    for epoch in range(num_epochs):
        # 前向傳播
        outputs = model(X_train)
        y_pred = torch.argmax(outputs,dim=1).to(torch.float)#.reshape(-1,1)
        # 計算損失
        loss = criterion(y_pred, y_train)
        total_loss += loss
        # 反向傳播
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # 更新參數
        optimizer.step()

    weights = model.linear.weight
    bias = model.linear.bias
    return model, weights, bias


def cross_feature_eval(alg, alg_type, criterion,X_train,cross_X_train,X_test,cross_X_test,y_train,y_test,num_epochs,learning_rate,model_eval):
    _, origin_weights, origin_bias = train(alg,
                                           alg_type,
                                           criterion, 
                                            X_train,
                                            y_train,
                                            num_epochs,
                                            learning_rate)
    _, cross_weights, cross_bias = train(alg, 
                                         alg_type,
                                         criterion, 
                                         cross_X_train,
                                         y_train,
                                         num_epochs,
                                         learning_rate)
    renew_weights = torch.cat((origin_weights,cross_weights),dim=1)
    renew_bias = torch.add(origin_bias,cross_bias)

    expanded_X_train = torch.cat((X_train,cross_X_train),dim=1)
    expanded_X_test = torch.cat((X_test,cross_X_test),dim=1)
    renew_model = alg(alg_type=alg_type,input_dim = expanded_X_train.shape[1], weights =renew_weights, bias = renew_bias)
    
    # 評估模型
    with torch.no_grad():
        # 前向傳播
        outputs = renew_model(expanded_X_test)
        y_pred = torch.argmax(outputs,dim=1).to(torch.float)#.reshape(-1,1)
        
        # 計算預測值
        if alg_type == "LogisticRegression":
            score = model_eval(y_pred.to(torch.int64), y_test.to(torch.int64))
        if alg_type == "LinearRegression":
            score = model_eval(y_pred, y_test)
    return score


def cross_feature_score(X_train,
                        y_train,
                        #X_test,
                        #y_test,
                        val_size,
                        random_state,
                        cross_index,
                        alg,
                        alg_type,
                        criterion,
                        num_epochs,
                        learning_rate,
                        model_eval):
        #print(cross_index)
        #print("HERE")
        X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_train, y_train,test_size=val_size,random_state=random_state)
        #print("HERE 2")
        cross_X_val_train = torch.prod(X_val_train[:,cross_index],dim=1).reshape(-1,1) # 假設長這樣，文獻維度也只有 1 
        cross_X_val_test = torch.prod(X_val_test[:,cross_index],dim=1).reshape(-1,1) # 假設長這樣，文獻維度也只有 1 

        score = cross_feature_eval(alg=alg, 
                                alg_type=alg_type,#"LinearRegression",#"LogisticRegression",
                                criterion=criterion,#nn.MSELoss(),#nn.BCELoss(),
                                X_train=X_val_train,
                                cross_X_train=cross_X_val_train,
                                X_test=X_val_test,
                                cross_X_test=cross_X_val_test,
                                y_train=y_val_train,
                                y_test=y_val_test,
                                num_epochs=num_epochs,
                                learning_rate=learning_rate,
                                model_eval=model_eval)
        #print(score.item())
        # 
        return score#.detach()#.item() # for CPU base on joblib

def combinations(iterable, r, remain_elements=None):
    # combinations('ABCD', 2) → AB AC AD BC BD CD
    # combinations(range(4), 3) → 012 013 023
    if remain_elements != None: 
        r = r - len(remain_elements) + 1
        iterable = list(map(str,iterable))
        remain_elements = list(map(str,remain_elements))
        iterable = set(iterable) - set(remain_elements)
        remain_elements = ";".join(remain_elements)
        iterable.add(remain_elements)
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    #print(tuple(pool[i] for i in indices))
    if (remain_elements != None) and (remain_elements in tuple(pool[i] for i in indices)):
        #yield tuple(pool[i] for i in indices)
        ans = list(pool[i] for i in indices)
        ans.remove(remain_elements)
        yield tuple(ans + remain_elements.split(";"))
    if remain_elements == None:
        yield tuple(pool[i] for i in indices)

    while True:
        for i in reversed(range(r)):
            #print(indices[i])
            if indices[i] != i + n - r:
                #print("break",indices[i])
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        if (remain_elements != None) and (remain_elements in tuple(pool[i] for i in indices)):
            ans = list(pool[i] for i in indices)
            ans.remove(remain_elements)
            yield tuple(ans + remain_elements.split(";"))
        if remain_elements == None:
            yield tuple(pool[i] for i in indices)
# drop_indices = [1,4,5]


def generate_combinations(ans, remain_elements, index=0, combination=None):
    if combination is None:
        combination = [0] * len(ans)

    if index == len(ans):
        yield tuple(combination)
        return

    if ";" in ans[index]: 
        for element in remain_elements:
            combination[index] = element
            yield from tuple(generate_combinations(ans, remain_elements, index + 1, combination))
    else:
        combination[index] = int(ans[index])
        yield from tuple(generate_combinations(ans, remain_elements, index + 1, combination))


def combinations_with_replacement(iterable, r, remain_elements=None):
    # combinations_with_replacement('ABC', 2) → AA AB AC BB BC CC
    if remain_elements != None: 
        r = r - len(remain_elements) + 1
        iterable = list(map(str,iterable))
        remain_elements = list(map(str,remain_elements))
        iterable = set(iterable) - set(remain_elements)
        remain_elements = ";".join(remain_elements)
        iterable.add(remain_elements)
    pool = tuple(iterable)
    n = len(pool)
    if not n and r:
        return
    indices = [0] * r
    if (remain_elements != None) and (remain_elements in tuple(pool[i] for i in indices)):
        ans = list(pool[i] for i in indices)
        ans.remove(remain_elements)
        # if remain_elements in ans:
        #     #print(ans)
        #     split_first = ans[0].split(';')
        #     # 使用列表解析生成结果列表
        #     for x in split_first:
        #         temp = [x] + ans[1:]
        #         #result.append(temp)
        #         yield tuple(temp)
        # else:
        combo_tuple = tuple(ans + remain_elements.split(";"))
        if any(map(lambda x: ";" in x,combo_tuple)):
            for gc in generate_combinations(list(combo_tuple), remain_elements.split(";")):
                yield tuple(map(int,gc))
        else:
            yield tuple(map(int,combo_tuple))
    if remain_elements == None:
        yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != n - 1:
                break
        else:
            return
        indices[i:] = [indices[i] + 1] * (r - i)


        if (remain_elements != None) and (remain_elements in tuple(pool[i] for i in indices)):
            ans = list(pool[i] for i in indices)
            ans.remove(remain_elements)
            # if remain_elements in ans:
            #     print(ans)
            #     split_first = ans[0].split(';')
            #     # 使用列表解析生成结果列表
            #     for x in split_first:
            #         temp = [x] + ans[1:]
            #         print(temp)
            #         #result.append(temp)
            #         yield tuple(temp)
            # else:
            combo_tuple = tuple(ans + remain_elements.split(";"))
            
            if any(map(lambda x: ";" in x,combo_tuple)):
                for gc in generate_combinations(list(combo_tuple), remain_elements.split(";")):
                    yield tuple(map(int,gc))
            else:
                yield tuple(map(int,combo_tuple))
        if remain_elements == None:
            yield tuple(pool[i] for i in indices)

def sampleSize(
    population_size,
    margin_error=.05,
    confidence_level=.99,
    sigma=1/2
):
    """
    Calculate the minimal sample size to use to achieve a certain
    margin of error and confidence level for a sample estimate
    of the population mean.

    Inputs
    -------
    population_size: integer
        Total size of the population that the sample is to be drawn from.

    margin_error: number
        Maximum expected difference between the true population parameter,
        such as the mean, and the sample estimate.

    confidence_level: number in the interval (0, 1)
        If we were to draw a large number of equal-size samples
        from the population, the true population parameter
        should lie within this percentage
        of the intervals (sample_parameter - e, sample_parameter + e)
        where e is the margin_error.

    sigma: number
        The standard deviation of the population.  For the case
        of estimating a parameter in the interval [0, 1], sigma=1/2
        should be sufficient.

    """
    alpha = 1 - (confidence_level)
    # dictionary of confidence levels and corresponding z-scores
    # computed via norm.ppf(1 - (alpha/2)), where norm is
    # a normal distribution object in scipy.stats.
    # Here, ppf is the percentile point function.
    zdict = {
        .90: 1.645,
        .91: 1.695,
        .99: 2.576,
        .97: 2.17,
        .94: 1.881,
        .93: 1.812,
        .95: 1.96,
        .98: 2.326,
        .96: 2.054,
        .92: 1.751
    }
    if confidence_level in zdict:
        z = zdict[confidence_level]
    else:
        from scipy.stats import norm
        z = norm.ppf(1 - (alpha/2))
    N = population_size
    M = margin_error
    numerator = z**2 * sigma**2 * (N / (N-1))
    denom = M**2 + ((z**2 * sigma**2)/(N-1))
    return numerator/denom


def compare(A,B):
    output = np.zeros((len(B), len(A)), dtype=int)

    # 將a中的元素映射到output矩陣中
    for i, row in enumerate(B):
        for j, item in enumerate(A):
            if item in row:
                output[i, j] = 1
    return output

class AutoFeatureCross:
    def __init__(self,
                 label_type,
                 val_size: float = 0.3,
                 num_epochs: int = 100,
                 learning_rate: float = 0.001,
                 combination_with_replacement: bool = True, # True, False
                 maximum_cross_number: Union[str, int]= "auto", # 'auto' # len(feature_name) 
                 searching_space: str = "auto",  # "sampling", "exhaustive", "auto" # 即使是exhaustive，也要
                 iteration:int = 30) -> None:

        self.val_size = val_size
        self.label_type = label_type        
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.combination_with_replacement = combination_with_replacement 
        self.maximum_cross_number = maximum_cross_number 
        self.searching_space = searching_space
        self.iteration = iteration
        
        if self.iteration < 30:
            raise Exception("number of iteration must greater than 30 ..") 
        if self.label_type == "classification":
            from torcheval.metrics.functional import multiclass_f1_score # , r2_score
            self.alg = Regression
            self.alg_type = "LogisticRegression"
            self.criterion = nn.BCELoss()
            self.model_eval = multiclass_f1_score
            self.direction = "maximum"
        elif self.label_type == "regression":
            from torcheval.metrics.functional import mean_squared_error
            self.alg = Regression
            self.alg_type = "LinearRegression"
            self.criterion = nn.MSELoss()
            self.model_eval = mean_squared_error
            self.direction = "minimum"
        else:
            raise Exception("please choose one of label type of question (clssification or regression)") 
        
        print(f" val_size: {self.val_size}\n",
              f"label_type: {self.label_type}\n", 
              f"num_epochs: {self.num_epochs}\n", 
              f"learning_rate: {self.learning_rate}\n", 
              f"combination_with_replacement: {self.combination_with_replacement}\n",
              f"maximum_cross_number: {self.maximum_cross_number}\n",
              f"searching_space: {self.searching_space}\n",
              f"iteration: {self.iteration} \n")
    def fit(self,X_train,y_train) -> None:

        if isinstance(X_train,pd.Series) or isinstance(X_train, pd.DataFrame) or isinstance(X_train, np.ndarray):
            X_train = torch.FloatTensor(X_train.to_numpy() if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series) else X_train).requires_grad_(True)
  
        if isinstance(y_train,pd.Series) or isinstance(y_train, pd.DataFrame) or isinstance(y_train, np.ndarray):
            y_train = torch.FloatTensor(y_train.to_numpy() if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series) else y_train).requires_grad_(True)

        recommand_cross_feature_set = set()
        recommand_cross_feature_list = list()

        if self.combination_with_replacement == True:
            combinations_method = combinations_with_replacement # combinations #combinations_with_replacement
        else:
            combinations_method = combinations
        feature_name = range(0,X_train.shape[1])
        # maximum_cross_number = "auto" #'auto' #'auto' # len(feature_name) 
        # 如果是combinations_with_replacement，強烈建議要調整，H74取8，就有300億個組合。光是找出組合組木就非常花時間
        # 而且 H74 也能取大於74以上，會算不完，要設置停損點
        # C75取8也要150億
        # 所以atuo的上限是針對取多少個做上限，目前是 上限 k = log2(feature_number) # 底數為 2是因為最少有2種組合，而feature_num > 4
        # maximum_cross_number --> 'auto' , 某某數字(>2)

        #searching_space = "auto" 

        #iteration = 30

        # best_score 有個瑕疵，就是會一定找到交互作用項目，但AUTOCROSS 則預設立場一定會有交互作用


        for epcho in tqdm(range(self.iteration)):
            print("-- EPCHO ",epcho+1, ' --')
            if self.direction == "maximum":
                best_score = -np.inf
            if self.direction == "minimum":
                best_score = np.inf
            
            cross_feature_info = dict()
            local_best_cross_feature = None
            best_cross_feature = None
            candidated_cross_feature_list = list()
            score_info = dict()
            
            k = 2
            
            while True:
                
                seed = np.random.randint(1,10000)

                if (self.maximum_cross_number if isinstance(self.maximum_cross_number, int) else False) and \
                (
                    (self.maximum_cross_number < 2) or \
                    (X_train.shape[1] < 2) or \
                    (self.maximum_cross_number > X_train.shape[1])
                ):
                    raise Exception(f"maximum_cross_number must be larger than 2 or less than {X_train.shape[1]}")
                

                for i in combinations_method(feature_name,k,best_cross_feature): # best_cross_feature # local_best_cross_feature
                    candidated_cross_feature_list.append(tuple(map(int,i)))
                
                if self.searching_space == "sampling":
                    # 接下來新增這個功能
                    # BEAM SEARCH 或 RANDOM SEARCH
                    sample_num = round(sampleSize(len(candidated_cross_feature_list)))
                    candidated_cross_feature_list = list(map(lambda x: f'{x}',candidated_cross_feature_list))
                    candidated_cross_feature_list = np.random.choice(candidated_cross_feature_list,size=sample_num)
                    candidated_cross_feature_list = list(map(lambda x: eval(x),candidated_cross_feature_list))        
                candidated_cross_feature_num = len(candidated_cross_feature_list)
                print("order:",k," candidated cross feature number: ",candidated_cross_feature_num)
                all_score = Parallel(n_jobs=n_jobs)(delayed(cross_feature_score)(X_train=X_train,
                                                                        y_train=y_train,
                                                                        val_size=self.val_size,
                                                                        random_state=seed,#,
                                                                        cross_index=i,
                                                                        alg=self.alg,
                                                                        alg_type=self.alg_type,
                                                                        criterion=self.criterion,# nn.MSELoss(),
                                                                        num_epochs=self.num_epochs,
                                                                        learning_rate=self.learning_rate,
                                                                        model_eval=self.model_eval) for i in candidated_cross_feature_list)
                
                for i,score in zip(candidated_cross_feature_list,all_score):
                    score_info[i] = score


                local_best_cross_feature= max(score_info, key=score_info.get) if self.direction == "maximum" else min(score_info, key=score_info.get)

                if (
                        (self.direction == "maximum") and \
                        (score_info[local_best_cross_feature] > best_score)
                    ) or \
                    (
                        (self.direction == "minimum") and \
                        (score_info[local_best_cross_feature] < best_score)
                    ):
                    
                    best_score = score_info[local_best_cross_feature]
                    
                    ## less computation but find less interaction
                    if best_cross_feature == None:
                        # 這是 H 作法
                        best_cross_feature = set(local_best_cross_feature) 
                        #以下是照著文獻做法，利用 C .set(local_best_cross_feature)
                    else:
                        # 這是 H的方法
                        best_cross_feature.update(set(local_best_cross_feature)) 
                    # print("best_cross_feature: ",best_cross_feature)
                    cross_feature_info[local_best_cross_feature]=best_score
                    breaking_step = 1 # 還是 1 # 如果是 1 就是痕容易重複把接下來把剩下的搜尋完成，算到
                else:
                    # breaking_step 這邊可能需要重新思考，因為可能會稀釋掉 HIGH ORDER的可能性
                    # 但因為時間關係需要拉扯
                    best_cross_feature_num= len(best_cross_feature)
                    breaking_step = best_cross_feature_num*(candidated_cross_feature_num)**(1/4)
                
                #cross_feature_info[local_best_cross_feature] = score_info[local_best_cross_feature]
                del score_info[local_best_cross_feature]
                
                # 因為效率問題，所以改成這樣， 有點類似 early stoping 的概念
                # 重要是這邊要如何定義出好的STOP
                # 目前發現高維度，越高ORDER可能只停留在 2 個交互作用或2個SET
                # 要如何自動化停止?
                if k > len(best_cross_feature):
                    stoping = k*breaking_step # 這邊是如果不斷找到相同的SET，沒有更好的SET，則期待預先停滯
                else :
                    stoping = k # 這邊則是找到新的SET後，重新計算步伐
        
            
                # 新增，讓K不繼續往上加，高維度無意義的找交互作用，但這是針對 H 的瑕疵
                # C 未測試
                if (self.maximum_cross_number == "auto") and (X_train.shape[1] >= 4):
                    k_stop = int(np.log2(X_train.shape[1])) # 用底數 2 是因為最小的交互作用
                else:
                    k_stop = self.maximum_cross_number
                step_stop = X_train.shape[1]

                k += 1
                if (
                    self.searching_space in ['auto','sampling'] and \
                        (
                            (stoping > step_stop) or (k > k_stop)
                        )
                    ) \
                    or \
                    (
                    self.searching_space in ['exhaustive'] and \
                        (
                            (k > step_stop) or (k  > self.maximum_cross_number if isinstance(self.maximum_cross_number, int) else False)
                        )
                    ): 
                    best_cross_feature_set = list(cross_feature_info.keys())
                    recommand_cross_feature_set.update(best_cross_feature_set)
                    recommand_cross_feature_list.append(best_cross_feature_set)
                    break
            self.recommand_cross_feature_set = recommand_cross_feature_set
            self.recommand_cross_feature_list = recommand_cross_feature_list
        return self
    
    @property
    def get_cross_feature_index(self):
        if self.recommand_cross_feature_list == list():
            print("there is no recommanded feature-cross ")
            return []
        else:
            from scipy.stats import mannwhitneyu
            from scipy.stats import false_discovery_control
            recommand_cross_feature_set = sorted(list(self.recommand_cross_feature_set)) # 要這要做，用set會BUG，list後排序會亂掉
            recommand_cross_feature_freq = compare(recommand_cross_feature_set,self.recommand_cross_feature_list)
            # 這不是好的統計方法，不太完全符合統計假說，理論上要用檢定哪個有很多零的發生，但目前沒有相關套件
            # 如果要自己刻和REVIEW，需要花很多時間
            # 但目前以下檢定方法是WORK的，雖然還是學術瑕疵
            recommand_cross_feature_freq_instance,recommand_cross_feature_freq_dim = recommand_cross_feature_freq.shape[0], recommand_cross_feature_freq.shape[1]
            reference = np.zeros(recommand_cross_feature_freq_instance, dtype=int)
            #reference = np.ones(recommand_cross_feature_freq.shape[0], dtype=int)
            pvalue_list = list()
            for i in range(recommand_cross_feature_freq_dim):
                pvalue = mannwhitneyu(recommand_cross_feature_freq[:,i],reference,alternative='greater').pvalue
                pvalue_list.append(pvalue)

            remain_recommand_cross_feature= np.array(list(map(lambda x: f'{x}',recommand_cross_feature_set)))[false_discovery_control(pvalue_list) < 0.05]
            remain_recommand_cross_feature = list(map(lambda x: eval(x),remain_recommand_cross_feature))
            self.remain_recommand_cross_feature = remain_recommand_cross_feature
            return remain_recommand_cross_feature
    
    
    def fit_transform(self,X):

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if not hasattr(self, 'remain_recommand_cross_feature'):
            self.get_cross_feature_index
            
        X = X.reset_index(drop=True)
        col_names = list(map(str,X.columns))
        cross_feature_info = {}
        for i in self.remain_recommand_cross_feature:
            feature_name = []
            data_cross = []
            for j in i:
                feature_name.append(col_names[j])
                data_cross.append(X.iloc[:,j])
            cross_feature_info["*".join(feature_name)] = np.prod(data_cross,axis=0)
        X = pd.concat([X,pd.DataFrame(cross_feature_info)],axis=1)
        return X
    


"""
autocross = AutoFeatureCross(
                      label_type = "regression", # regression, classification
                      num_epochs = 150,
                      learning_rate = 0.002,
                      combination_with_replacement = True, # False
                      maximum_cross_number = 2, #"auto" 2, 3, ....
                      searching_space = "auto", # sampling, exhaustive, auto
                      iteration = 30
                      )

autocross.fit(X_train, y_train)
autocross.fit_transform(X_train)
"""