import shutil

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import dill
import xlrd


high_identity=['1STN','1EY0']
low_identity=['1AMQ','1APS','1ARR','1B26','1BTA','1BVC','1CAH','1CEY','1CHK','1CYO','1FC1','1FEP','1FKJ','1H7M','1HK0','1HNG','1HUE','1HZ6','1IHB','1IO2','1IR3','1KFW','1LVE','1N0J','1PGA','1PIN','1QJP','1QLP','1QND','1QQV','1RBP','1RHG','1RIS','1RN1','1ROP','1SAK','1SSO','1THQ','1TIT','1TYV','1UZC','1VQB','1ZNJ','2ABD','2ADA','2AKY','2BRD','2CRK','2NVH','2Q98','2RN2','2TRX','3ECA','3MBP','3PGK','3SSI','451C','5CRO']

# low identity pdb protein-level cv
data = pd.read_csv("./data/fea_data/S7089_fea_after_double.csv")  # 读取数据
rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X

BO_params= pd.read_excel("./resource/pro_BO_Best_Param.xlsx")


sc = StandardScaler()  # 定义标准化模型
sc.fit(X)  # 标准化

count=0
mse_list = []
rmse_list = []
mae_list = []
r2_list = []
pearson_list = []
spearman_list = []

who_pdb=sorted(list(set(low_identity)),reverse=True)
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
for train_pdb_idx, test_pdb_idx in kfold.split(who_pdb):
    train_pdb=[who_pdb[id] for id in train_pdb_idx]
    test_pdb=[who_pdb[id] for id in test_pdb_idx]
    condition = data['ID'].str.split('_').str[0].isin(train_pdb)
    train_data = data[condition]
    condition = data['ID'].str.split('_').str[0].isin(test_pdb)
    test_data = data[condition]
    train_data.drop('ID',axis=1)
    test_data.drop('ID',axis=1)
    X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()
    X_train=train_data[X_cols].values
    X_test=test_data[X_cols].values
    y_train=train_data['Experimental_DDG'].values
    y_test=test_data['Experimental_DDG'].values
    X_train=sc.transform(X_train)
    X_test=sc.transform(X_test)
    model = XGBRegressor(n_estimators=int(BO_params['n_estimators']),max_depth=int(BO_params['max_depth']),eta=float(BO_params['eta']),subsample=float(BO_params['subsample']),colsample_bytree=float(BO_params['colsample_bytree']),learning_rate=float(BO_params['learning_rate']),random_state=42)  # 定义XGBoost模型
    model.fit(X_train, y_train)  # 训练模型
    y_pred = model.predict(X_test)  # 预测
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    y_test = np.array(y_test).reshape((-1, 1))
    y_pred = y_pred.reshape((-1, 1))

    yy = np.concatenate([y_test, y_pred], -1)
    yy = yy.T
    corr_matrix = np.corrcoef(yy)
    pearson = corr_matrix[0][1]
    correlation, p_value = spearmanr(y_test, y_pred)
    spearman = correlation

    print(f'cross {count}:')
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R^2 Score:", r2)
    print("pearson:", pearson)
    print("spearman:", spearman)

    mse_list.append(mse)
    rmse_list.append(rmse)
    mae_list.append(mae)
    r2_list.append(r2)
    pearson_list.append(pearson)
    spearman_list.append(spearman)

    count+=1

print('average: ')
print("MSE:", sum(mse_list) / 10)
print("RMSE:", sum(rmse_list) / 10)
print("MAE:", sum(mae_list) / 10)
print("R^2 Score:", sum(r2_list) / 10)
print("pearson:", sum(pearson_list) / 10)
print("spearman:", sum(spearman_list) / 10)



evaluate_result = pd.DataFrame(
    np.array([mse_list, rmse_list, mae_list, r2_list, pearson_list, spearman_list]),
    index=["mse", "rmse", "mae", "r2", "pearson", "spearman"],
    columns=[f"split{i}_test_score" for i in range(10)],
)
evaluate_result["mean"] = evaluate_result.mean(axis=1)  # 计算平均评估结果
evaluate_result["std"] = evaluate_result.std(axis=1)  # 计算评估结果标准差
evaluate_result.to_excel("./evaluation/low_identity_pro_cv_10fold.xlsx", index=True)  # 保存评估结果


# average
# 均方误差（MSE）: 4.109878494114311
# 均方根误差（RMSE）: 1.957124435891983
# 平均绝对误差（MAE）: 1.3310843097544796
# 决定系数（R^2 Score）: 0.09874141555540705
# 皮尔逊相关系数（pearson）: 0.33525904683036106
# 斯皮尔曼相关系数（spearman）: 0.3912503282150138



# high identity pdb HRMs-level cv
data = pd.read_csv("./data/fea_data/S7089_fea_after_double.csv")  # 读取数据
# condition = data['ID'].str.split('_').str[0].str.contains('1STN')
condition = data['ID'].str.split('_').str[0].isin(high_identity)
data = data[condition]

rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X
y = data["Experimental_DDG"].values  # 取出目标值 y

sc = StandardScaler()  # 定义标准化模型
X = sc.fit_transform(X)  # 标准化

BO_params= pd.read_excel("./resource/BO_Best_Param.xlsx")




count=0
mse_list = []
rmse_list = []
mae_list = []
r2_list = []
pearson_list = []
spearman_list = []




groups = [i // 2 for i in range(len(X))]  # define groups
for train_idxs, test_idxs in GroupKFold(n_splits=10).split(X, groups=groups):  # 10折交叉验证
    count+=1
    X_train, X_test = X[train_idxs], X[test_idxs]  # 划分训练集和测试集
    y_train, y_test = y[train_idxs], y[test_idxs]  # 划分训练集和测试集
    model = XGBRegressor(n_estimators=int(BO_params['n_estimators']),max_depth=int(BO_params['max_depth']),eta=float(BO_params['eta']),subsample=float(BO_params['subsample']),colsample_bytree=float(BO_params['colsample_bytree']),learning_rate=float(BO_params['learning_rate']),random_state=42)  # 定义XGBoost模型
    model.fit(X_train, y_train)  # 训练模型
    y_pred = model.predict(X_test)  # 预测

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    y_test = np.array(y_test).reshape((-1, 1))
    y_pred = y_pred.reshape((-1, 1))
    yy = np.concatenate([y_test, y_pred], -1)
    yy = yy.T
    corr_matrix = np.corrcoef(yy)
    pearson = corr_matrix[0][1]

    correlation, p_value = spearmanr(y_test, y_pred)
    spearman = correlation

    print(f'cross {count}:')
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R^2 Score:", r2)
    print("pearson:", pearson)
    print("spearman:", spearman)

    mse_list.append(mse)
    rmse_list.append(rmse)
    mae_list.append(mae)
    r2_list.append(r2)
    pearson_list.append(pearson)
    spearman_list.append(spearman)





print('average: ')
print("MSE:", sum(mse_list) / 10)
print("RMSE:", sum(rmse_list) / 10)
print("MAE:", sum(mae_list) / 10)
print("R^2 Score:", sum(r2_list) / 10)
print("pearson:", sum(pearson_list) / 10)
print("spearman:", sum(spearman_list) / 10)


evaluate_result = pd.DataFrame(
    np.array([mse_list, rmse_list, mae_list, r2_list, pearson_list, spearman_list]),
    index=["mse", "rmse", "mae", "r2", "pearson", "spearman"],
    columns=[f"split{i}_test_score" for i in range(10)],
)
evaluate_result["mean"] = evaluate_result.mean(axis=1)  # 计算平均评估结果
evaluate_result["std"] = evaluate_result.std(axis=1)  # 计算评估结果标准差
evaluate_result.to_excel("./evaluation/high_identity_cv_10fold.xlsx", index=True)  # 保存评估结果


# average
# 均方误差（MSE）: 0.622670199002943
# 均方根误差（RMSE）: 0.777801994503666
# 平均绝对误差（MAE）: 0.5135820102477611
# 决定系数（R^2 Score）: 0.879405416555942
# 皮尔逊相关系数（pearson）: 0.9384821627320081
# 斯皮尔曼相关系数（spearman）: 0.9253977106799292






