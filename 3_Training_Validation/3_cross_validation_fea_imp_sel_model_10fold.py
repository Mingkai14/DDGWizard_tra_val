import shutil
import dill
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def save_pkl(filepath, data):  # save data (model) as pkl format to filepath
    with open(filepath, "wb") as fw:
        dill.dump(data, fw)
    print(f"[{filepath}] data saving...")

if not os.path.exists('./resource/DDGWizard/'):
    os.mkdir('./resource/DDGWizard/')


data = pd.read_csv("./data/fea_data/S7089_fea_after_double.csv")  # 读取数据
data = data.drop("ID", axis=1)  # 删除ID列

rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X
y = data["Experimental_DDG"].values  # 取出目标值 y

sc = StandardScaler()  # 定义标准化模型
X = sc.fit_transform(X)  # 标准化
save_pkl('./resource/DDGWizard/sc.pkl',sc)

BO_params= pd.read_excel("./resource/BO_Best_Param.xlsx")



count=0
mse_list = []
rmse_list = []
mae_list = []
r2_list = []
pearson_list = []
spearman_list = []
feature_importances_recorder = []

best_r2=0.0
best_model=None

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
    feature_importances_recorder.append(model.feature_importances_)

    if r2>best_r2:
        best_r2=r2
        best_model=model



feature_importances = pd.DataFrame(
    np.array(feature_importances_recorder).T,
    index=X_cols,
    columns=[f"split{i}_test_score" for i in range(10)],
)  # 特征重要性
feature_importances["mean"] = feature_importances.mean(axis=1)  # 计算平均特征重要性
feature_importances.to_excel("./evaluation/feature_importances.xlsx", index=True)  # 保存特征重要性
feature_importances["mean"].sort_values(ascending=False).to_excel("./evaluation/feature_importances_ranked.xlsx", index=True)



print('average: ')
print("MSE:", sum(mse_list) / 10)
print("RMSE:", sum(rmse_list) / 10)
print("MAE:", sum(mae_list) / 10)
print("R^2 Score:", sum(r2_list) / 10)
print("pearson:", sum(pearson_list) / 10)
print("spearman:", sum(spearman_list) / 10)


print('best r2:')
print(best_r2)
save_pkl("./resource/DDGWizard/predictor.pkl", best_model)
shutil.copyfile('./resource/rfe_infos.xlsx','./resource/DDGWizard/rfe_infos.xlsx')



# average
# 均方误差（MSE）: 1.601192128736796
# 均方根误差（RMSE）: 1.264104711442045
# 平均绝对误差（MAE）: 0.7622761167506115
# 决定系数（R^2 Score）: 0.6050646863032705
# 皮尔逊相关系数（pearson）: 0.779393060682583
# 斯皮尔曼相关系数（spearman）: 0.7450787650858806


