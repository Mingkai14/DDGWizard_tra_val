import shutil

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evalue(true_for:list,pred_for:list,true_rev:list,pred_rev:list,true_total:list,pred_total:list):
    for_mse = mean_squared_error(true_for, pred_for)
    for_rmse = np.sqrt(mean_squared_error(true_for, pred_for))
    for_mae = mean_absolute_error(true_for, pred_for)
    for_r2 = r2_score(true_for, pred_for)

    y_test = np.array(true_for).reshape((-1, 1))
    y_pred = np.array(pred_for).reshape((-1, 1))
    yy = np.concatenate([y_test, y_pred], -1)
    yy = yy.T
    corr_matrix = np.corrcoef(yy)
    for_pearson = corr_matrix[0][1]

    correlation, p_value = spearmanr(y_test, y_pred)
    for_spearman = correlation

    rev_mse = mean_squared_error(true_rev, pred_rev)
    rev_rmse = np.sqrt(mean_squared_error(true_rev, pred_rev))
    rev_mae = mean_absolute_error(true_rev, pred_rev)
    rev_r2 = r2_score(true_rev, pred_rev)

    y_test = np.array(true_rev).reshape((-1, 1))
    y_pred = np.array(pred_rev).reshape((-1, 1))
    yy = np.concatenate([y_test, y_pred], -1)
    yy = yy.T
    corr_matrix = np.corrcoef(yy)
    rev_pearson = corr_matrix[0][1]

    correlation, p_value = spearmanr(y_test, y_pred)
    rev_spearman = correlation

    total_mse = mean_squared_error(true_total, pred_total)
    total_rmse = np.sqrt(mean_squared_error(true_total, pred_total))
    total_mae = mean_absolute_error(true_total, pred_total)
    total_r2 = r2_score(true_total, pred_total)

    y_test = np.array(true_total).reshape((-1, 1))
    y_pred = np.array(pred_total).reshape((-1, 1))
    yy = np.concatenate([y_test, y_pred], -1)
    yy = yy.T
    corr_matrix = np.corrcoef(yy)
    total_pearson = corr_matrix[0][1]

    correlation, p_value = spearmanr(y_test, y_pred)
    total_spearman = correlation

    covariance = np.cov(pred_for, pred_rev)[0, 1]
    std_deviation_forward = np.std(pred_for)
    std_deviation_reverse = np.std(pred_rev)
    r_dr = covariance / (std_deviation_forward * std_deviation_reverse)

    assert len(pred_for) == len(pred_rev)
    count = len(pred_for)
    sum = 0.0
    for i in range(count):
        sum += pred_for[i] + pred_rev[i]
    bias = sum / (count * 2)

    return for_pearson,for_r2,for_rmse,rev_pearson,rev_r2,rev_rmse,total_pearson,total_r2,total_rmse,r_dr,bias



data = pd.read_csv("./data/fea_data/S7089_fea_after_double.csv")  # 读取数据
data = data.drop("ID", axis=1)  # 删除ID列






rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X
y = data["Experimental_DDG"].values  # 取出目标值 y
temp_test=y.tolist()


sc = StandardScaler()  # 定义标准化模型
X = sc.fit_transform(X)  # 标准化

BO_params= pd.read_excel("./resource/BO_Best_Param.xlsx")



count=1
for_pearson_list=[]
for_r2_list=[]
for_rmse_list=[]
rev_pearson_list=[]
rev_r2_list=[]
rev_rmse_list=[]
total_pearson_list=[]
total_r2_list=[]
total_rmse_list=[]
r_dr_list=[]
bias_list=[]

whole_y_test=np.array([])
whole_y_pred=np.array([])

each_fold_true_data_DDGWizard=[]
each_fold_pred_data_DDGWizard=[]

groups = [i // 2 for i in range(len(X))]  # define groups
for train_idxs, test_idxs in GroupKFold(n_splits=20).split(X, groups=groups):  # 10折交叉验证
    count+=1
    X_train, X_test = X[train_idxs], X[test_idxs]  # 划分训练集和测试集
    y_train, y_test = y[train_idxs], y[test_idxs]  # 划分训练集和测试集

    model = XGBRegressor(n_estimators=int(BO_params['n_estimators']),max_depth=int(BO_params['max_depth']),eta=float(BO_params['eta']),subsample=float(BO_params['subsample']),colsample_bytree=float(BO_params['colsample_bytree']),learning_rate=float(BO_params['learning_rate']),random_state=42)  # 定义XGBoost模型
    model.fit(X_train, y_train)  # 训练模型
    y_pred = model.predict(X_test)  # 预测

    whole_y_test=np.append(whole_y_test,y_test)
    whole_y_pred=np.append(whole_y_pred, y_pred)

    true_total=y_test.tolist()
    pred_total=y_pred.tolist()
    true_for=true_total[::2]
    true_rev=true_total[1::2]
    pred_for=pred_total[::2]
    pred_rev=pred_total[1::2]

    each_fold_true_data_DDGWizard.append(true_total)
    each_fold_pred_data_DDGWizard.append(pred_total)

    assert len(true_total) % 2 == 0
    assert len(pred_total) % 2 == 0

    for_pearson,for_r2,for_rmse,rev_pearson,rev_r2,rev_rmse,total_pearson,total_r2,total_rmse,r_dr,bias=evalue(true_for,pred_for,true_rev,pred_rev,true_total,pred_total)
    for_pearson_list.append(for_pearson)
    for_r2_list.append(for_r2)
    for_rmse_list.append(for_rmse)
    rev_pearson_list.append(rev_pearson)
    rev_r2_list.append(rev_r2)
    rev_rmse_list.append(rev_rmse)
    total_pearson_list.append(total_pearson)
    total_r2_list.append(total_r2)
    total_rmse_list.append(total_rmse)
    r_dr_list.append(r_dr)
    bias_list.append(bias)


print(sum(for_pearson_list) / 20)
print(sum(for_r2_list) / 20)
print(sum(for_rmse_list) / 20)
print(sum(rev_pearson_list) / 20)
print(sum(rev_r2_list) / 20)
print(sum(rev_rmse_list) / 20)
print(sum(total_pearson_list) / 20)
print(sum(total_r2_list) / 20)
print(sum(total_rmse_list) / 20)
print(sum(r_dr_list) / 20)
print(sum(bias_list) / 20)

import xlwt

columns=['forward_pearson','forward_RMSE','reverse_pearson','reverse_RMSE','total_pearson','total_RMSE','r_dr','bias']
output=[sum(for_pearson_list) / 20,sum(for_rmse_list) / 20,sum(rev_pearson_list) / 20,sum(rev_rmse_list) / 20,sum(total_pearson_list) / 20,sum(total_rmse_list) / 20,sum(r_dr_list) / 20,sum(bias_list) / 20]
wb=xlwt.Workbook()
ws=wb.add_sheet('sheet1')
for i in range(len(columns)):
    ws.write(0,i,columns[i])
for i in range(len(output)):
    ws.write(1,i,output[i])
wb.save('./evaluation/cv_20fold_DDGWizard.xls')

ddgun3d_data=pd.read_excel("./data/ddgun_res.xls")
ddgun3d_data=ddgun3d_data.drop("id",axis=1)
ddgun3d_data=ddgun3d_data.drop("for_or_rev",axis=1)
ddgun3d_data=ddgun3d_data.drop("pdb_path",axis=1)
ddgun3d_data=ddgun3d_data.drop("mutation",axis=1)
ddgun3d_data=ddgun3d_data.drop("chain",axis=1)
y_ddgun3d=ddgun3d_data["ddg"].values



true_total = temp_test
pred_total = y_ddgun3d
true_for = true_total[::2]
true_rev = true_total[1::2]
pred_for = pred_total[::2]
pred_rev = pred_total[1::2]
for_pearson,for_r2,for_rmse,rev_pearson,rev_r2,rev_rmse,total_pearson,total_r2,total_rmse,r_dr,bias=evalue(true_for,pred_for,true_rev,pred_rev,true_total,pred_total)
print(for_pearson)
print(for_r2)
print(for_rmse)
print(rev_pearson)
print(rev_r2)
print(rev_rmse)
print(total_pearson)
print(total_r2)
print(total_rmse)
print(r_dr)
print(bias)

columns=['forward_pearson','forward_RMSE','reverse_pearson','reverse_RMSE','total_pearson','total_RMSE','r_dr','bias']
output=[for_pearson,for_rmse,rev_pearson,rev_rmse,total_pearson,total_rmse,r_dr,bias]
wb=xlwt.Workbook()
ws=wb.add_sheet('sheet1')
for i in range(len(columns)):
    ws.write(0,i,columns[i])
for i in range(len(output)):
    ws.write(1,i,output[i])
wb.save('./evaluation/cv_20fold_ddgun3D.xls')



acdc_data=pd.read_excel("./data/acdcnn_res.xls")
acdc_data=acdc_data.drop("id",axis=1)
acdc_data=acdc_data.drop("for_or_rev",axis=1)
acdc_data=acdc_data.drop("pdb_path",axis=1)
acdc_data=acdc_data.drop("mutation",axis=1)
acdc_data=acdc_data.drop("chain",axis=1)
y_acdc=acdc_data["ddg"].values

true_total = temp_test
pred_total = y_acdc
true_for = true_total[::2]
true_rev = true_total[1::2]
pred_for = pred_total[::2]
pred_rev = pred_total[1::2]
for_pearson,for_r2,for_rmse,rev_pearson,rev_r2,rev_rmse,total_pearson,total_r2,total_rmse,r_dr,bias=evalue(true_for,pred_for,true_rev,pred_rev,true_total,pred_total)
print(for_pearson)
print(for_r2)
print(for_rmse)
print(rev_pearson)
print(rev_r2)
print(rev_rmse)
print(total_pearson)
print(total_r2)
print(total_rmse)
print(r_dr)
print(bias)

columns=['forward_pearson','forward_RMSE','reverse_pearson','reverse_RMSE','total_pearson','total_RMSE','r_dr','bias']
output=[for_pearson,for_rmse,rev_pearson,rev_rmse,total_pearson,total_rmse,r_dr,bias]
wb=xlwt.Workbook()
ws=wb.add_sheet('sheet1')
for i in range(len(columns)):
    ws.write(0,i,columns[i])
for i in range(len(output)):
    ws.write(1,i,output[i])
wb.save('./evaluation/cv_20fold_acdc.xls')

scatter_dict={'x':whole_y_test,'y':whole_y_pred}
scatter_df=pd.DataFrame(scatter_dict)
scatter_df.to_excel("./evaluation/scatter_result.xlsx", index=False, header=True)

assert len(each_fold_true_data_DDGWizard)==20
assert len(each_fold_pred_data_DDGWizard)==20

data = []

# 遍历每个子列表的索引
for i in range(len(each_fold_true_data_DDGWizard)):
    # 添加标记行
    data.append([f"Number {i + 1} fold"])

    # 添加两个子列表中的数据
    for j in range(len(each_fold_true_data_DDGWizard[i])):
        data.append([each_fold_true_data_DDGWizard[i][j], each_fold_pred_data_DDGWizard[i][j]])

# 将数据转换为DataFrame
df = pd.DataFrame(data, columns=["True", "Pred"])

# 将DataFrame保存到Excel文件中
df.to_excel("./data/DDGWizard_res.xlsx", index=False, header=False)

