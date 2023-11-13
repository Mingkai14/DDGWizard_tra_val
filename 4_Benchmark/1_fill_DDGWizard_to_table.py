import dill
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def load_pkl(filepath):  # load model of pkl format from filepath, and return data (model)
    with open(filepath, "rb") as fr:
        data = dill.load(fr, encoding="utf-8")
    print(f"[{filepath}] data loading...")
    return data


#forward
data = pd.read_csv("./data/fea_data/S787_fea_only_forward.csv")  # 读取数据
data = data.drop("ID", axis=1)  # 删除ID列

rfe_infos = pd.read_excel("./resource/DDGWizard/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X

sc=load_pkl('./resource/DDGWizard/sc.pkl')
X = sc.transform(X)  # 标准化

model=load_pkl('./resource/DDGWizard/predictor.pkl')
y_pred = model.predict(X)
pred_ddg_forward=list(y_pred)

#reverse
data = pd.read_csv("./data/fea_data/S787_fea_only_reverse.csv")  # 读取数据
data = data.drop("ID", axis=1)  # 删除ID列

rfe_infos = pd.read_excel("./resource/DDGWizard/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X

sc=load_pkl('./resource/DDGWizard/sc.pkl')
X = sc.transform(X)  # 标准化

model=load_pkl('./resource/DDGWizard/predictor.pkl')
y_pred = model.predict(X)
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


