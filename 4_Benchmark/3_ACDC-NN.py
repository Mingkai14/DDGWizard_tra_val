from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import xlrd


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
test_res=read_xls('./data/tes_res.xls')

true_for=[]
true_rev=[]
pred_for=[]
pred_rev=[]
true_total=[]
pred_total=[]

for res in test_res:
    if res[15]=='' or res[16]=='':
        true_for.append(res[7])
        true_rev.append(res[8])
        pred_for.append(res[7])
        pred_rev.append(res[8])
        true_total.append(res[7])
        true_total.append(res[8])
        pred_total.append(res[7])
        pred_total.append(res[8])
    else:
        true_for.append(res[7])
        true_rev.append(res[8])
        pred_for.append(res[15])
        pred_rev.append(res[16])
        true_total.append(res[7])
        true_total.append(res[8])
        pred_total.append(res[15])
        pred_total.append(res[16])


output=[]
columns=['forward_pearson','forward_RMSE','reverse_pearson','reverse_RMSE','total_pearson','total_RMSE','r_dr','bias','forward_correct_sign','forward_correct_sign','inconsistence']

#forward

mse = mean_squared_error(true_for, pred_for)
rmse = np.sqrt(mean_squared_error(true_for, pred_for))
mae = mean_absolute_error(true_for, pred_for)
r2 = r2_score(true_for, pred_for)

y_test = np.array(true_for).reshape((-1, 1))
y_pred = np.array(pred_for).reshape((-1, 1))
yy = np.concatenate([y_test, y_pred], -1)
yy = yy.T
corr_matrix = np.corrcoef(yy)
pearson = corr_matrix[0][1]

correlation, p_value = spearmanr(y_test, y_pred)
spearman = correlation

print('forward:')
print(mse)
print(rmse)
print(mae)
print(r2)
print(pearson)
print(spearman)
print(float(p_value))

output.append(pearson)
output.append(rmse)

# MSE: 2.8600971636537458
# RMSE: 1.691182179321242
# MAE: 1.1234242987642948
# R^2 Score: 0.2091829212678772
# pearson: 0.47675838198160275
# spearman: 0.4406524542533614
# p-value: 1.0176228950711751e-38



# reverse

mse = mean_squared_error(true_rev, pred_rev)
rmse = np.sqrt(mean_squared_error(true_rev, pred_rev))
mae = mean_absolute_error(true_rev, pred_rev)
r2 = r2_score(true_rev, pred_rev)

y_test = np.array(true_rev).reshape((-1, 1))
y_pred = np.array(pred_rev).reshape((-1, 1))
yy = np.concatenate([y_test, y_pred], -1)
yy = yy.T
corr_matrix = np.corrcoef(yy)
pearson = corr_matrix[0][1]

correlation, p_value = spearmanr(y_test, y_pred)
spearman = correlation

print('reverse:')
print(mse)
print(rmse)
print(mae)
print(r2)
print(pearson)
print(spearman)
print(float(p_value))

output.append(pearson)
output.append(rmse)

# MSE: 2.8600971636537458
# RMSE: 1.691182179321242
# MAE: 1.1234242987642948
# R^2 Score: 0.2091829212678772
# pearson: 0.47675838198160275
# spearman: 0.4406524542533614
# p-value: 1.0176228950711751e-38


# total

mse = mean_squared_error(true_total, pred_total)
rmse = np.sqrt(mean_squared_error(true_total, pred_total))
mae = mean_absolute_error(true_total, pred_total)
r2 = r2_score(true_total, pred_total)

y_test = np.array(true_total).reshape((-1, 1))
y_pred = np.array(pred_total).reshape((-1, 1))
yy = np.concatenate([y_test, y_pred], -1)
yy = yy.T
corr_matrix = np.corrcoef(yy)
pearson = corr_matrix[0][1]

correlation, p_value = spearmanr(y_test, y_pred)
spearman = correlation

print('total:')
print(mse)
print(rmse)
print(mae)
print(r2)
print(pearson)
print(spearman)
print(float(p_value))

output.append(pearson)
output.append(rmse)

# MSE: 2.8600971636537458
# RMSE: 1.691182179321242
# MAE: 1.1234242987642948
# R^2 Score: 0.28295671312930504
# pearson: 0.5385487794888746
# spearman: 0.5159765971317362
# p-value: 8.308441495803925e-108

covariance = np.cov(pred_for, pred_rev)[0, 1]
std_deviation_forward = np.std(pred_for)
std_deviation_reverse = np.std(pred_rev)
r_dr=covariance/(std_deviation_forward*std_deviation_reverse)
print('r_dr:')
print(r_dr)
output.append(r_dr)
#-1.0012722646310435

assert len(pred_for)==len(pred_rev)
count=len(pred_for)
sum=0.0
for i in range(count):
    sum+=pred_for[i]+pred_rev[i]
bias=sum/(count*2)
print('bias:')
print(bias)
output.append(bias)
#0

assert len(true_for)==len(pred_for)
assert len(true_rev)==len(pred_rev)

sign_correctly_prediction_forward=0
count=len(true_for)
for i in range(count):
    if true_for[i]*pred_for[i]>0:
        sign_correctly_prediction_forward+=1
sign_correctly_predicted_forward=sign_correctly_prediction_forward/count
print('sign_correctly_predicted_forward:')
print(sign_correctly_predicted_forward)
output.append(sign_correctly_predicted_forward)
#0.6683608640406608

sign_correctly_prediction_reverse=0
count=len(true_rev)
for i in range(count):
    if true_rev[i]*pred_rev[i]>0:
        sign_correctly_prediction_reverse+=1
sign_correctly_predicted_reverse=sign_correctly_prediction_reverse/count
print('sign_correctly_predicted_reverse:')
print(sign_correctly_predicted_reverse)
output.append(sign_correctly_predicted_reverse)
#0.6683608640406608

assert len(pred_for)==len(pred_rev)
count=len(pred_for)
inconsistent_count=0
for i in range(count):
    if pred_for[i]*pred_rev[i]>0:
        inconsistent_count+=1
inconsistence=inconsistent_count/count
print('inconsistence:')
print(inconsistence)
output.append(inconsistence)
# 0

import xlwt
wb=xlwt.Workbook()
ws=wb.add_sheet('sheet1')
for i in range(len(columns)):
    ws.write(0,i,columns[i])
for i in range(len(output)):
    ws.write(1,i,output[i])
wb.save('./evaluation/acdc_nn_tes_res.xls')

