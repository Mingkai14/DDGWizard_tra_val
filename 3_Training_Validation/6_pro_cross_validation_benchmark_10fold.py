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

data = pd.read_csv("./data/fea_data/S7089_fea_after_double.csv")  # 读取数据
rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X

BO_params= pd.read_excel("./resource/pro_BO_Best_Param.xlsx")


sc = StandardScaler()  # 定义标准化模型
sc.fit(X)  # 标准化

ddgun3d_data=pd.read_excel("./data/ddgun_res.xls")
ddgun3d_data=ddgun3d_data.drop("for_or_rev",axis=1)
ddgun3d_data=ddgun3d_data.drop("pdb_path",axis=1)
ddgun3d_data=ddgun3d_data.drop("mutation",axis=1)
ddgun3d_data=ddgun3d_data.drop("chain",axis=1)

acdc_data=pd.read_excel("./data/acdcnn_res.xls")
acdc_data=acdc_data.drop("for_or_rev",axis=1)
acdc_data=acdc_data.drop("pdb_path",axis=1)
acdc_data=acdc_data.drop("mutation",axis=1)
acdc_data=acdc_data.drop("chain",axis=1)




def read_xls(name):
    rb = xlrd.open_workbook(name)
    rs=rb.sheet_by_index(0)
    row = rs.nrows
    col = rs.ncols
    xls_list=[]
    for i in range(1,row):
        row_list=[]
        for j in range(col):
            row_list.append(rs.cell_value(i,j))
        xls_list.append(row_list)
    return xls_list

raw_set=read_xls('./data/raw_data/S7089.xls')

count=1
mse_list = []
rmse_list = []
mae_list = []
r2_list = []
pearson_list = []
spearman_list = []

mse_list_ddgun3d = []
rmse_list_ddgun3d = []
mae_list_ddgun3d = []
r2_list_ddgun3d = []
pearson_list_ddgun3d = []
spearman_list_ddgun3d = []

mse_list_acdc = []
rmse_list_acdc = []
mae_list_acdc = []
r2_list_acdc = []
pearson_list_acdc = []
spearman_list_acdc = []

who_pdb=[]
for row in raw_set:
    who_pdb.append(row[0])
who_pdb=sorted(list(set(who_pdb)),reverse=True)
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
for train_pdb_idx, test_pdb_idx in kfold.split(who_pdb):
    train_pdb=[who_pdb[id] for id in train_pdb_idx]
    test_pdb=[who_pdb[id] for id in test_pdb_idx]
    condition = data['ID'].str.split('_').str[0].isin(train_pdb)
    train_data = data[condition]
    condition = data['ID'].str.split('_').str[0].isin(test_pdb)
    test_data = data[condition]
    condition = ddgun3d_data['id'].str.split('_').str[0].isin(test_pdb)
    test_data_ddgun3d=ddgun3d_data[condition]
    condition = acdc_data['id'].str.split('_').str[0].isin(test_pdb)
    test_data_acdc=acdc_data[condition]

    train_data.drop('ID',axis=1)
    test_data.drop('ID',axis=1)
    test_data_ddgun3d.drop('id',axis=1)
    test_data_acdc.drop('id', axis=1)
    rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")
    X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()
    X_train=train_data[X_cols].values
    X_test=test_data[X_cols].values
    y_train=train_data['Experimental_DDG'].values
    y_test=test_data['Experimental_DDG'].values
    y_pred_ddgun3d=test_data_ddgun3d['ddg'].values
    y_pred_acdc = test_data_acdc['ddg'].values

    X_train=sc.transform(X_train)
    X_test=sc.transform(X_test)

    model = XGBRegressor(n_estimators=int(BO_params['n_estimators']),max_depth=int(BO_params['max_depth']),eta=float(BO_params['eta']),subsample=float(BO_params['subsample']),colsample_bytree=float(BO_params['colsample_bytree']),learning_rate=float(BO_params['learning_rate']),random_state=42)  # 定义XGBoost模型
    model.fit(X_train, y_train)  # 训练模型
    y_pred = model.predict(X_test)  # 预测

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse_ddgun3d = mean_squared_error(y_test, y_pred_ddgun3d)
    rmse_ddgun3d = np.sqrt(mean_squared_error(y_test, y_pred_ddgun3d))
    mae_ddgun3d = mean_absolute_error(y_test, y_pred_ddgun3d)
    r2_ddgun3d = r2_score(y_test, y_pred_ddgun3d)

    mse_acdc = mean_squared_error(y_test, y_pred_acdc)
    rmse_acdc = np.sqrt(mean_squared_error(y_test, y_pred_acdc))
    mae_acdc = mean_absolute_error(y_test, y_pred_acdc)
    r2_acdc = r2_score(y_test, y_pred_acdc)

    y_test = np.array(y_test).reshape((-1, 1))
    y_pred = y_pred.reshape((-1, 1))
    y_pred_ddgun3d = y_pred_ddgun3d.reshape((-1, 1))
    y_pred_acdc = y_pred_acdc.reshape((-1, 1))
    yy = np.concatenate([y_test, y_pred], -1)
    yy = yy.T
    corr_matrix = np.corrcoef(yy)
    pearson = corr_matrix[0][1]

    yy = np.concatenate([y_test, y_pred_ddgun3d], -1)
    yy = yy.T
    corr_matrix = np.corrcoef(yy)
    pearson_ddgun3d = corr_matrix[0][1]

    yy = np.concatenate([y_test, y_pred_acdc], -1)
    yy = yy.T
    corr_matrix = np.corrcoef(yy)
    pearson_acdc = corr_matrix[0][1]

    correlation, p_value = spearmanr(y_test, y_pred)
    spearman = correlation

    correlation, p_value = spearmanr(y_test, y_pred_ddgun3d)
    spearman_ddgun3d = correlation

    correlation, p_value = spearmanr(y_test, y_pred_acdc)
    spearman_acdc = correlation



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


    print(f'ddgun3d:')
    print("MSE:", mse_ddgun3d)
    print("RMSE:", rmse_ddgun3d)
    print("MAE:", mae_ddgun3d)
    print("R^2 Score:", r2_ddgun3d)
    print("pearson:", pearson_ddgun3d)
    print("spearman:", spearman_ddgun3d)

    mse_list_ddgun3d.append(mse_ddgun3d)
    rmse_list_ddgun3d.append(rmse_ddgun3d)
    mae_list_ddgun3d.append(mae_ddgun3d)
    r2_list_ddgun3d.append(r2_ddgun3d)
    pearson_list_ddgun3d.append(pearson_ddgun3d)
    spearman_list_ddgun3d.append(spearman_ddgun3d)


    print(f'acdc:')
    print("MSE:", mse_acdc)
    print("RMSE:", rmse_acdc)
    print("MAE:", mae_acdc)
    print("R^2 Score:", r2_acdc)
    print("pearson:", pearson_acdc)
    print("spearman:", spearman_acdc)

    mse_list_acdc.append(mse_acdc)
    rmse_list_acdc.append(rmse_acdc)
    mae_list_acdc.append(mae_acdc)
    r2_list_acdc.append(r2_acdc)
    pearson_list_acdc.append(pearson_acdc)
    spearman_list_acdc.append(spearman_acdc)

    count+=1




print('average: ')
print("MSE:", sum(mse_list) / 10)
print("RMSE:", sum(rmse_list) / 10)
print("MAE:", sum(mae_list) / 10)
print("R^2 Score:", sum(r2_list) / 10)
print("pearson:", sum(pearson_list) / 10)
print("spearman:", sum(spearman_list) / 10)

print("MSE:", sum(mse_list_ddgun3d) / 10)
print("RMSE:", sum(rmse_list_ddgun3d) / 10)
print("MAE:", sum(mae_list_ddgun3d) / 10)
print("R^2 Score:", sum(r2_list_ddgun3d) / 10)
print("pearson:", sum(pearson_list_ddgun3d) / 10)
print("spearman:", sum(spearman_list_ddgun3d) / 10)

print("MSE:", sum(mse_list_acdc) / 10)
print("RMSE:", sum(rmse_list_acdc) / 10)
print("MAE:", sum(mae_list_acdc) / 10)
print("R^2 Score:", sum(r2_list_acdc) / 10)
print("pearson:", sum(pearson_list_acdc) / 10)
print("spearman:", sum(spearman_list_acdc) / 10)





evaluate_result = pd.DataFrame(
    np.array([mse_list, rmse_list, mae_list, r2_list, pearson_list, spearman_list]),
    index=["mse", "rmse", "mae", "r2", "pearson", "spearman"],
    columns=[f"split{i}_test_score" for i in range(10)],
)
evaluate_result["mean"] = evaluate_result.mean(axis=1)  # 计算平均评估结果
evaluate_result["std"] = evaluate_result.std(axis=1)  # 计算评估结果标准差
evaluate_result.to_excel("./evaluation/pro_cv_10fold_DDGWizard.xlsx", index=True)  # 保存评估结果



evaluate_result = pd.DataFrame(
    np.array([mse_list_ddgun3d, rmse_list_ddgun3d, mae_list_ddgun3d, r2_list_ddgun3d, pearson_list_ddgun3d, spearman_list_ddgun3d]),
    index=["mse", "rmse", "mae", "r2", "pearson", "spearman"],
    columns=[f"split{i}_test_score" for i in range(10)],
)
evaluate_result["mean"] = evaluate_result.mean(axis=1)  # 计算平均评估结果
evaluate_result["std"] = evaluate_result.std(axis=1)  # 计算评估结果标准差
evaluate_result.to_excel("./evaluation/pro_cv_10fold_ddgun3D.xlsx", index=True)  # 保存评估结果



evaluate_result = pd.DataFrame(
    np.array([mse_list_acdc, rmse_list_acdc, mae_list_acdc, r2_list_acdc, pearson_list_acdc, spearman_list_acdc]),
    index=["mse", "rmse", "mae", "r2", "pearson", "spearman"],
    columns=[f"split{i}_test_score" for i in range(10)],
)
evaluate_result["mean"] = evaluate_result.mean(axis=1)  # 计算平均评估结果
evaluate_result["std"] = evaluate_result.std(axis=1)  # 计算评估结果标准差
evaluate_result.to_excel("./evaluation/pro_cv_10fold_acdc.xlsx", index=True)  # 保存评估结果




# average
# DDGWizard
# 均方误差（MSE）: 2.9851468433924606
# 均方根误差（RMSE）: 1.7211756554246396
# 平均绝对误差（MAE）: 1.1467699157816105
# 决定系数（R^2 Score）: 0.22341105794392296
# 皮尔逊相关系数（pearson）: 0.4920940237990667
# 斯皮尔曼相关系数（spearman）: 0.5163512040703052

# ddgun3D
# 均方误差（MSE）: 3.276456190819984
# 均方根误差（RMSE）: 1.8020860145842739
# 平均绝对误差（MAE）: 1.216343533856484
# 决定系数（R^2 Score）: 0.14078760955139463
# 皮尔逊相关系数（pearson）: 0.45173020160196814
# 斯皮尔曼相关系数（spearman）: 0.44902913664323263

# acdcnn
# 均方误差（MSE）: 2.9458140821949765
# 均方根误差（RMSE）: 1.7110634705916088
# 平均绝对误差（MAE）: 1.1217318905295033
# 决定系数（R^2 Score）: 0.2273328607824716
# 皮尔逊相关系数（pearson）: 0.49481922382410276
# 斯皮尔曼相关系数（spearman）: 0.5194445260291175