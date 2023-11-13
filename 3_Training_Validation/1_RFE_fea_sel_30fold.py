import dill
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def save_pkl(filepath, data):  # save data (model) as pkl format to filepath
    with open(filepath, "wb") as fw:
        dill.dump(data, fw)
    print(f"[{filepath}] data saving...")


def load_pkl(filepath):  # load model of pkl format from filepath, and return data (model)
    with open(filepath, "rb") as fr:
        data = dill.load(fr, encoding="utf-8")
    print(f"[{filepath}] data loading...")
    return data


if __name__ == "__main__":
    data = pd.read_csv("./data/fea_data/S7089_fea_after_double.csv")  # read fearures_table
    data = data.drop("ID", axis=1)  # drop ID column
    X = data.drop(columns=["Experimental_DDG", "Experimental_DDG_Classification"], axis=1).copy()  # drop target value
    y = data["Experimental_DDG"].tolist()  # obtain target value

    sc = StandardScaler()  # Standardize
    X[X.columns] = sc.fit_transform(X[X.columns].values)

    groups = [i // 2 for i in range(len(X))]  # define groups

    model = XGBRegressor(random_state=42)  # define XGBoost model
    rfecv = RFECV(                         # define RFE model
        estimator=model,
        step=1,
        min_features_to_select=1,  # set minimum of features as 1
        cv=GroupKFold(n_splits=30),  # 30-fold cross-validation
        scoring="r2",  # use R-squared as score metrics
        verbose=100,
        n_jobs=-1,
    )
    rfecv.fit(X, y, groups=groups)  # train rfe to select features
    save_pkl("./resource/rfecv.pkl", rfecv)  # save rfe model


    rfecv = load_pkl("./resource/rfecv.pkl")  # load rfe model

    print("features num: ", rfecv.n_features_in_)  # 总特征数量
    print("best features num: ", rfecv.n_features_)  # 最佳特征数量

    rfe_infos = pd.DataFrame(
        {
            "feature_names": rfecv.feature_names_in_,
            "ranking": rfecv.ranking_,
            "support": rfecv.support_,
        }
    )  # RFE模型的特征信息
    rfe_infos.to_excel("./resource/rfe_infos.xlsx", index=False)  # 保存RFE模型的特征信息

    rfe_results = pd.DataFrame(rfecv.cv_results_)  # RFE模型的结果
    rfe_results.to_excel("./evaluation/rfe_results.xlsx", index=False)  # 保存RFE模型的结果

    rfe_results_mean = pd.DataFrame(rfecv.cv_results_["mean_test_score"])
    rfe_results_mean.to_excel("./evaluation/rfe_results_mean.xlsx", index=True, header=False)