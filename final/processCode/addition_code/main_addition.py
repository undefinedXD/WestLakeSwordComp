# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
# import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import math
from data_preprocessing import feature_engineering

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from utils import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import lightgbm as lgb
import joblib
from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def addtion_func(addtion_path):
    train_data = pd.read_csv("../data/train.csv")
    train_ = pd.read_csv("train_初赛数据.csv")
    train_data = train_data.append(train_).reset_index(drop=True)
    train_data = train_data[train_data.label==1]

    train_data = feature_engineering(train_data.copy())
    train_data.drop(['eventId'],inplace=True,axis=1)

    add = pd.read_csv(addtion_path)
    add = add[add.label==1]
    test = pd.read_csv("../data/test_1.csv")
    add_ = test[test.eventId.isin(add.eventId)]
    add_ = feature_engineering(add_.copy())

    cat_features = ['tlsSni', 'tlsVersion', 'country','state','organization','OU','CN','issueCountry','issueState','issueOrganization','issueOU','issueCN','srcAddress','destAddress']


    target = train_data.pop("label")
    target_enc = TargetEncoder(cols=cat_features)
    target_enc.fit(train_data[cat_features],target)
    train_data[cat_features] = target_enc.transform(train_data[cat_features])
    add_[cat_features] =  target_enc.transform(add_[cat_features])

    train_data['label'] = target
    train_data  = train_data[train_data.label==1]
    target  = train_data.pop("label")

    eventId = add_.pop("eventId")
    kk = train_data.isnull().sum()
    dd = add_.isnull().sum()

    for item in list(dd[dd!=0].index):
        add_[item] = add_[item].fillna(-1)


    for item in list(kk[kk!=0].index):
        train_data[item] = train_data[item].fillna(-1)

    lens = add_.shape[0]

    data = train_data.append(add_).reset_index(drop=True)
    # 计算DBSCAN
    db = DBSCAN(eps=20, min_samples=10).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # 聚类的结果
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    labels_pred = labels[-lens:]
    print(set(labels_pred))

    pred_df = pd.Series(labels_pred, name='label', index=eventId).reset_index()
    pred_df.to_csv("./result.csv",index=False)

if __name__ == '__main__':
    addtion_path = '../data/addtion.csv' # 该路径仅供参考
    addtion_func(addtion_path)
