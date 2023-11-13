import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xlrd

data = pd.read_csv("./data/fea_data/S7089_fea_after_double.csv")  # 读取数据
rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X

sc = StandardScaler()  # 定义标准化模型
X = sc.fit(X)  # 标准化


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

who_pdb=[]
for row in raw_set:
    who_pdb.append(row[0])
who_pdb=sorted(list(set(who_pdb)),reverse=True)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)


def xgb_cv(n_estimators,max_depth,eta,subsample,colsample_bytree,learning_rate):
    params={
        'n_estimators': 10,
        'max_depth': 1,
        'eta': 0.01,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'learning_rate': 0.1
        }
    params.update({'n_estimators':int(n_estimators),'max_depth':int(max_depth),'eta':eta,'subsample':subsample,'colsample_bytree':colsample_bytree,'learning_rate':learning_rate})

    r2_list = []
    for train_pdb_idx, test_pdb_idx in kfold.split(who_pdb):
        train_pdb = [who_pdb[id] for id in train_pdb_idx]
        test_pdb = [who_pdb[id] for id in test_pdb_idx]
        condition = data['ID'].str.split('_').str[0].isin(train_pdb)
        train_data = data[condition]
        condition = data['ID'].str.split('_').str[0].isin(test_pdb)
        test_data = data[condition]

        train_data.drop('ID', axis=1)
        test_data.drop('ID', axis=1)
        rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")
        X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()
        X_train = train_data[X_cols].values
        X_test = test_data[X_cols].values
        y_train = train_data['Experimental_DDG'].values
        y_test = test_data['Experimental_DDG'].values
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_list.append(r2)

    return sum(r2_list) / 10


param_value_dics={
                   'n_estimators':(10, 1000),
                    'max_depth':(1, 10),
                   'eta':(0.01,1),
                   'subsample':(0.1, 1.0),
                   'colsample_bytree':(0.1, 1.0),
                   'learning_rate': (0.001,0.1)
               }




# 建立贝叶斯调参对象，迭代500次
lgb_bo = BayesianOptimization(
        xgb_cv,
        param_value_dics,
        random_state=42
    )
lgb_bo.maximize(init_points=5,n_iter=100) #init_points-调参基准点，n_iter-迭代次数

# 查看最优参数结果
print(lgb_bo.max)
# 查看全部调参结果
print(lgb_bo.res)




best_param = pd.DataFrame(lgb_bo.max['params'], index=['values'])
best_param.to_excel("./resource/pro_BO_Best_Param.xlsx", index=True)










