import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold






#load data and drop ID column
df = pd.read_csv('./data/S7089_fea_after_double.csv')
df.drop('ID', axis=1, inplace=True)

fea_index_class_a=[0,1]
fea_index_class_b=[2,720]
fea_index_class_c=[721,1374]
fea_index_class_d=[1375,1520]
fea_index_class_e=[1521,1546]

names_class_a=[]
names_class_b=[]
names_class_c=[]
names_class_d=[]
names_class_e=[]

table_names = list(df.columns)

for i in range(fea_index_class_a[0],fea_index_class_a[1]+1):
    names_class_a.append(table_names[i])
for i in range(fea_index_class_b[0],fea_index_class_b[1]+1):
    names_class_b.append(table_names[i])
for i in range(fea_index_class_c[0],fea_index_class_c[1]+1):
    names_class_c.append(table_names[i])
for i in range(fea_index_class_d[0],fea_index_class_d[1]+1):
    names_class_d.append(table_names[i])
for i in range(fea_index_class_e[0],fea_index_class_e[1]+1):
    names_class_e.append(table_names[i])


basic_class_dict={'a':names_class_a,'b':names_class_b,'c':names_class_c,'d':names_class_d,'e':names_class_e}
class_dict={}

#layer1
for key in basic_class_dict.keys():
    class_dict[key]=basic_class_dict[key]
#layer2
for key1 in basic_class_dict.keys():
    for key2 in basic_class_dict.keys():
        order_list=sorted([str(key1),str(key2)])
        temp_key=''.join(order_list)
        if key1==key2 or temp_key in class_dict.keys():
            continue
        temp_list=basic_class_dict[key1]+basic_class_dict[key2]
        class_dict[temp_key]=temp_list
#layer3
for key1 in basic_class_dict.keys():
    for key2 in basic_class_dict.keys():
        for key3 in basic_class_dict.keys():
            order_list = sorted([str(key1), str(key2),str(key3)])
            temp_key = ''.join(order_list)
            if key1==key2 or key1==key3 or key2==key3 or temp_key in class_dict.keys():
                continue
            temp_list=basic_class_dict[key1]+basic_class_dict[key2]+basic_class_dict[key3]
            class_dict[temp_key]=temp_list
#layer4
for key1 in basic_class_dict.keys():
    for key2 in basic_class_dict.keys():
        for key3 in basic_class_dict.keys():
            for key4 in basic_class_dict.keys():
                order_list = sorted([str(key1), str(key2), str(key3),str(key4)])
                temp_key = ''.join(order_list)
                if key1==key2 or key1==key3 or key1==key4 or key2==key3 or key2==key4 or key3==key4 or temp_key in class_dict.keys():
                    continue
                temp_list=basic_class_dict[key1]+basic_class_dict[key2]+basic_class_dict[key3]+basic_class_dict[key4]
                class_dict[temp_key]=temp_list
#layer5
for key1 in basic_class_dict.keys():
    for key2 in basic_class_dict.keys():
        for key3 in basic_class_dict.keys():
            for key4 in basic_class_dict.keys():
                for key5 in basic_class_dict.keys():
                    order_list = sorted([str(key1), str(key2), str(key3),str(key4),str(key5)])
                    temp_key = ''.join(order_list)
                    if key1==key2 or key1==key3 or key1==key4 or key1==key5 or key2==key3 or key2==key4 or key2==key5 or key3==key4 or key3==key5 or key4==key5 or temp_key in class_dict.keys():
                        continue
                    temp_list=basic_class_dict[key1]+basic_class_dict[key2]+basic_class_dict[key3]+basic_class_dict[key4]
                    class_dict[temp_key]=temp_list


def Compute(df,table_names:list,order:str):
    table_names = table_names + ['Experimental_DDG', 'Experimental_DDG_Classification']
    df = df.loc[:, table_names]

    Y_reg = list(df['Experimental_DDG'])
    Y_cls = list(df['Experimental_DDG_Classification'])

    df.drop(['Experimental_DDG', 'Experimental_DDG_Classification'], axis=1, inplace=True)
    fea = df.values

    sc = StandardScaler()
    fea = sc.fit_transform(fea)

    X = np.array(fea)
    y = np.array(Y_reg)

    groups = [i // 2 for i in range(len(X))]

    kfold = GroupKFold(n_splits=30)

    count = 0
    mse_list = []
    rmse_list = []
    mae_list = []
    r2_list = []
    pearson_list = []
    spearman_list = []

    for train_idx, test_idx in kfold.split(X,groups=groups):
        count += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # create model and train
        model = XGBRegressor()

        model.fit(X_train, y_train)

        #predict to obtain prediction value
        y_pred = model.predict(X_test)

        #evaluate
        r2 = r2_score(y_test, y_pred)

        r2_list.append(r2)


    mean_r2=sum(r2_list)/30

    return [order,mean_r2]

res_dict={}
for key in class_dict:
    res=Compute(df,class_dict[key],str(key))
    res_dict[res[0]]=res[1]
print(res_dict)
res_df=pd.DataFrame(res_dict,index=[0]).T
res_df.to_excel("./evaluation/5Type_Groups_R2.xlsx", index=True, header=False)




total_dict={}
count_dict={}
mean_dict={}
for key in res_dict.keys():
    s_len= len(str(key))
    if 'layer'+str(s_len) not in total_dict.keys():
        total_dict['layer'+str(s_len)]=0
    total_dict['layer'+str(s_len)]+=res_dict[key]
    if 'layer'+str(s_len) not in count_dict.keys():
        count_dict['layer'+str(s_len)]=0
    count_dict['layer'+str(s_len)]+=1
for key in total_dict.keys():
    mean_dict[key]=total_dict[key]/count_dict[key]
print(mean_dict)
mean_df=pd.DataFrame(mean_dict,index=[0]).T
mean_df.to_excel("./evaluation/5Layer_Mean_R2.xlsx", index=True, header=False)






