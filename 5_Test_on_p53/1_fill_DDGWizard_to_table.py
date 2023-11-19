import shutil
import dill
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_validate, GroupKFold
import os





data = pd.read_csv("./data/fea_data/S7089_fea_after_double.csv")  # 读取数据
condition = ~data['ID'].str.split('_').str[0].str.contains('2OCJ')
data = data[condition]
data = data.drop("ID", axis=1)  # 删除ID列

rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X
y = data["Experimental_DDG"].values  # 取出目标值 y

sc = StandardScaler()  # 定义标准化模型
X = sc.fit_transform(X)  # 标准化


BO_params= pd.read_excel("./resource/BO_Best_Param.xlsx")



best_r2=0.0
best_model=None

groups = [i // 2 for i in range(len(X))]  # define groups
for train_idxs, test_idxs in GroupKFold(n_splits=20).split(X, groups=groups):  # 10折交叉验证
    X_train, X_test = X[train_idxs], X[test_idxs]  # 划分训练集和测试集
    y_train, y_test = y[train_idxs], y[test_idxs]  # 划分训练集和测试集
    model = XGBRegressor(n_estimators=int(BO_params['n_estimators']),max_depth=int(BO_params['max_depth']),eta=float(BO_params['eta']),subsample=float(BO_params['subsample']),colsample_bytree=float(BO_params['colsample_bytree']),learning_rate=float(BO_params['learning_rate']),random_state=42)  # 定义XGBoost模型
    model.fit(X_train, y_train)  # 训练模型
    y_pred = model.predict(X_test)  # 预测

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)



    if r2>best_r2:
        best_r2=r2
        best_model=model







#forward
data = pd.read_csv("./data/fea_data/p53_fea_only_forward.csv")  # 读取数据
data = data.drop("ID", axis=1)  # 删除ID列

rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X

X = sc.transform(X)  # 标准化


y_pred = best_model.predict(X)
pred_ddg_forward=list(y_pred)

#reverse
data = pd.read_csv("./data/fea_data/p53_fea_only_reverse.csv")  # 读取数据
data = data.drop("ID", axis=1)  # 删除ID列

rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X

X = sc.transform(X)  # 标准化

y_pred = best_model.predict(X)
pred_ddg_reverse=list(y_pred)

assert len(pred_ddg_forward)==len(pred_ddg_reverse)
import xlrd
from xlutils.copy import copy
rb=xlrd.open_workbook('./data/tes_res.xls')
wb=copy(rb)
ws=wb.get_sheet(0)


for i in range(len(pred_ddg_forward)):
    ws.write(i+1,13,float(pred_ddg_forward[i]))
    ws.write(i+1,14,float(pred_ddg_reverse[i]))

wb.save('./data/tes_res.xls')





