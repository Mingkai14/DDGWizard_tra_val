import shutil

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score




data = pd.read_csv("./data/fea_data/S7089_fea_after_double.csv")  # 读取数据
data = data.drop("ID", axis=1)  # 删除ID列

ddgun3d_data=pd.read_excel("./data/ddgun_res.xls")
ddgun3d_data=ddgun3d_data.drop("id",axis=1)
ddgun3d_data=ddgun3d_data.drop("for_or_rev",axis=1)
ddgun3d_data=ddgun3d_data.drop("pdb_path",axis=1)
ddgun3d_data=ddgun3d_data.drop("mutation",axis=1)
ddgun3d_data=ddgun3d_data.drop("chain",axis=1)
y_ddgun3d=ddgun3d_data["ddg"].values

acdc_data=pd.read_excel("./data/acdcnn_res.xls")
acdc_data=acdc_data.drop("id",axis=1)
acdc_data=acdc_data.drop("for_or_rev",axis=1)
acdc_data=acdc_data.drop("pdb_path",axis=1)
acdc_data=acdc_data.drop("mutation",axis=1)
acdc_data=acdc_data.drop("chain",axis=1)
y_acdc=acdc_data["ddg"].values


rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X
y = data["Experimental_DDG"].values  # 取出目标值 y

sc = StandardScaler()  # 定义标准化模型
X = sc.fit_transform(X)  # 标准化

BO_params= pd.read_excel("./resource/BO_Best_Param.xlsx")



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

whole_y_test=np.array([])
whole_y_pred=np.array([])

groups = [i // 2 for i in range(len(X))]  # define groups
for train_idxs, test_idxs in GroupKFold(n_splits=10).split(X, groups=groups):  # 10折交叉验证
    count+=1
    X_train, X_test = X[train_idxs], X[test_idxs]  # 划分训练集和测试集
    y_train, y_test = y[train_idxs], y[test_idxs]  # 划分训练集和测试集
    y_pred_ddgun3d=y_ddgun3d[test_idxs]
    y_pred_acdc = y_acdc[test_idxs]

    model = XGBRegressor(n_estimators=int(BO_params['n_estimators']),max_depth=int(BO_params['max_depth']),eta=float(BO_params['eta']),subsample=float(BO_params['subsample']),colsample_bytree=float(BO_params['colsample_bytree']),learning_rate=float(BO_params['learning_rate']),random_state=42)  # 定义XGBoost模型
    model.fit(X_train, y_train)  # 训练模型
    y_pred = model.predict(X_test)  # 预测

    whole_y_test=np.append(whole_y_test,y_test)
    whole_y_pred=np.append(whole_y_pred, y_pred)

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
    print('DDGWizard:')
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R^2 Score:", r2)
    print("pearson:", pearson)
    print("spearman:", spearman)
    print('\n')

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
    print('\n')

    mse_list_ddgun3d.append(mse_ddgun3d)
    rmse_list_ddgun3d.append(rmse_ddgun3d)
    mae_list_ddgun3d.append(mae_ddgun3d)
    r2_list_ddgun3d.append(r2_ddgun3d)
    pearson_list_ddgun3d.append(pearson_ddgun3d)
    spearman_list_ddgun3d.append(spearman_ddgun3d)

    print(f'acdcnn:')
    print("MSE:", mse_acdc)
    print("RMSE:", rmse_acdc)
    print("MAE:", mae_acdc)
    print("R^2 Score:", r2_acdc)
    print("pearson:", pearson_acdc)
    print("spearman:", spearman_acdc)
    print('\n')

    mse_list_acdc.append(mse_acdc)
    rmse_list_acdc.append(rmse_acdc)
    mae_list_acdc.append(mae_acdc)
    r2_list_acdc.append(r2_acdc)
    pearson_list_acdc.append(pearson_acdc)
    spearman_list_acdc.append(spearman_acdc)







print('average: ')
print('DDGWizard')
print("MSE:", sum(mse_list) / 10)
print("RMSE:", sum(rmse_list) / 10)
print("MAE:", sum(mae_list) / 10)
print("R^2 Score:", sum(r2_list) / 10)
print("pearson:", sum(pearson_list) / 10)
print("spearman:", sum(spearman_list) / 10)
print('\n')


print('ddgun3D:')
print("MSE:", sum(mse_list_ddgun3d) / 10)
print("RMSE:", sum(rmse_list_ddgun3d) / 10)
print("MAE:", sum(mae_list_ddgun3d) / 10)
print("R^2 Score:", sum(r2_list_ddgun3d) / 10)
print("pearson:", sum(pearson_list_ddgun3d) / 10)
print("spearman:", sum(spearman_list_ddgun3d) / 10)
print('\n')


print('acdcnn:')
print("MSE:", sum(mse_list_acdc) / 10)
print("RMSE:", sum(rmse_list_acdc) / 10)
print("MAE:", sum(mae_list_acdc) / 10)
print("R^2 Score:", sum(r2_list_acdc) / 10)
print("pearson:", sum(pearson_list_acdc) / 10)
print("spearman:", sum(spearman_list_acdc) / 10)
print('\n')




evaluate_result = pd.DataFrame(
    np.array([mse_list, rmse_list, mae_list, r2_list, pearson_list, spearman_list]),
    index=["mse", "rmse", "mae", "r2", "pearson", "spearman"],
    columns=[f"split{i}_test_score" for i in range(10)],
)
evaluate_result["mean"] = evaluate_result.mean(axis=1)  # 计算平均评估结果
evaluate_result["std"] = evaluate_result.std(axis=1)  # 计算评估结果标准差
evaluate_result.to_excel("./evaluation/cv_10fold_DDGWizard.xlsx", index=True)  # 保存评估结果



evaluate_result = pd.DataFrame(
    np.array([mse_list_ddgun3d, rmse_list_ddgun3d, mae_list_ddgun3d, r2_list_ddgun3d, pearson_list_ddgun3d, spearman_list_ddgun3d]),
    index=["mse", "rmse", "mae", "r2", "pearson", "spearman"],
    columns=[f"split{i}_test_score" for i in range(10)],
)
evaluate_result["mean"] = evaluate_result.mean(axis=1)  # 计算平均评估结果
evaluate_result["std"] = evaluate_result.std(axis=1)  # 计算评估结果标准差
evaluate_result.to_excel("./evaluation/cv_10fold_ddgun3D.xlsx", index=True)  # 保存评估结果



evaluate_result = pd.DataFrame(
    np.array([mse_list_acdc, rmse_list_acdc, mae_list_acdc, r2_list_acdc, pearson_list_acdc, spearman_list_acdc]),
    index=["mse", "rmse", "mae", "r2", "pearson", "spearman"],
    columns=[f"split{i}_test_score" for i in range(10)],
)
evaluate_result["mean"] = evaluate_result.mean(axis=1)  # 计算平均评估结果
evaluate_result["std"] = evaluate_result.std(axis=1)  # 计算评估结果标准差
evaluate_result.to_excel("./evaluation/cv_10fold_acdc.xlsx", index=True)  # 保存评估结果

scatter_dict={'x':whole_y_test,'y':whole_y_pred}
scatter_df=pd.DataFrame(scatter_dict)
scatter_df.to_excel("./evaluation/scatter_result.xlsx", index=False, header=True)



# average
# DDGWizard
# 均方误差（MSE）: 1.601192128736796
# 均方根误差（RMSE）: 1.264104711442045
# 平均绝对误差（MAE）: 0.7622761167506115
# 决定系数（R^2 Score）: 0.6050646863032705
# 皮尔逊相关系数（pearson）: 0.779393060682583
# 斯皮尔曼相关系数（spearman）: 0.7450787650858806

# ddgun3D
# 均方误差（MSE）: 3.114536332769158
# 均方根误差（RMSE）: 1.7609841826561958
# 平均绝对误差（MAE）: 1.2142918991497536
# 决定系数（R^2 Score）: 0.2344887593298505
# 皮尔逊相关系数（pearson）: 0.5005516568160708
# 斯皮尔曼相关系数（spearman）: 0.46253219527135175

# acdcnn
# 均方误差（MSE）: 2.8738503029883455
# 均方根误差（RMSE）: 1.691926216522333
# 平均绝对误差（MAE）: 1.1425670170814561
# 决定系数（R^2 Score）: 0.29379758694924946
# 皮尔逊相关系数（pearson）: 0.5422484744543478
# 斯皮尔曼相关系数（spearman）: 0.5366823268681301