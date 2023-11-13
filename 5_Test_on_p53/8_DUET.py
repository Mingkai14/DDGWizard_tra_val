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
    if res[23]=='' or res[24]=='':
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
        pred_for.append(res[23])
        pred_rev.append(res[24])
        true_total.append(res[7])
        true_total.append(res[8])
        pred_total.append(res[23])
        pred_total.append(res[24])


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

# MSE: 1.7069326904761903
# RMSE: 1.3064963415471895
# MAE: 1.041452380952381
# R^2 Score: 0.46716727339579656
# pearson: 0.726462143017056
# spearman: 0.6595356757571088
# p-value: 2.0302898954096448e-06



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

# MSE: 7.362237928571429
# RMSE: 2.7133444176092776
# MAE: 2.0548333333333333
# R^2 Score: -1.2981816045102663
# pearson: 0.02024375706918
# spearman: -0.022284348302396475
# p-value: 0.888598366514586


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

# MSE: 4.53458530952381
# RMSE: 2.129456576106639
# MAE: 1.5481428571428573
# R^2 Score: 0.13656745676435478
# pearson: 0.5117073696631362
# spearman: 0.40972900743636703
# p-value: 0.0001086891755990259

covariance = np.cov(pred_for, pred_rev)[0, 1]
std_deviation_forward = np.std(pred_for)
std_deviation_reverse = np.std(pred_rev)
r_dr=covariance/(std_deviation_forward*std_deviation_reverse)
print('r_dr:')
print(r_dr)
output.append(r_dr)
#-0.16286457789928066

assert len(pred_for)==len(pred_rev)
count=len(pred_for)
sum=0.0
for i in range(count):
    sum+=pred_for[i]+pred_rev[i]
bias=sum/(count*2)
print('bias:')
print(bias)
output.append(bias)
# -0.7849999999999999

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
#0.7619047619047619

sign_correctly_prediction_reverse=0
count=len(true_rev)
for i in range(count):
    if true_rev[i]*pred_rev[i]>0:
        sign_correctly_prediction_reverse+=1
sign_correctly_predicted_reverse=sign_correctly_prediction_reverse/count
print('sign_correctly_predicted_reverse:')
print(sign_correctly_predicted_reverse)
output.append(sign_correctly_predicted_reverse)
#0.47619047619047616

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
# 0.6190476190476191

import xlwt
wb=xlwt.Workbook()
ws=wb.add_sheet('sheet1')
for i in range(len(columns)):
    ws.write(0,i,columns[i])
for i in range(len(output)):
    ws.write(1,i,output[i])
wb.save('./evaluation/DUET_tes_res.xls')

