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



def train_func(train_path):
#请填写训练代码
    train_data = pd.read_csv(train_path)
    train_features_selected = feature_engineering(train_data)

    cat_features = ['tlsSni', 'tlsVersion', 'country','state','organization','OU','CN','issueCountry','issueState','issueOrganization',
                    'issueOU','issueCN','srcAddress','destAddress']

    kf = StratifiedKFold(n_splits=5 ,shuffle=True, random_state=42)

    target = train_features_selected.pop("label")
    train_features_selected.drop("eventId",inplace=True,axis=1)

    for i, (train_index, valid_index) in enumerate(kf.split(train_features_selected, target)):
        X_train_split, y_train_split, X_val, y_val = train_features_selected.iloc[train_index], target[train_index], train_features_selected.iloc[valid_index], target[valid_index]
        
        target_enc = TargetEncoder(cols=cat_features)
        target_enc.fit(X_train_split[cat_features],y_train_split)
        
        X_train_split = X_train_split.join(target_enc.transform(X_train_split[cat_features]).add_suffix('_target'))
        X_val = X_val.join(target_enc.transform(X_val[cat_features]).add_suffix('_target'))

        X_train_split.drop(cat_features,inplace=True,axis=1)
        X_val.drop(cat_features,inplace=True,axis=1)
        
        train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
        valid_matrix = lgb.Dataset(X_val, label=y_val)

        base_params_lgb = {
                            'boosting_type': 'gbdt',
                            'objective': 'binary',
                            'metric': 'binary_logloss',
                            'learning_rate': 0.001,
                            'num_leaves': 82,
                            'max_depth': 8,
    #                         'num_leaves': 49,
    #                         'max_depth': 6,
                            'min_data_in_leaf': 64,
                            'min_child_weight':1.435,
                            'bagging_fraction': 0.785,
                            'feature_fraction': 0.373,
                            'bagging_freq': 22,
                            'reg_lambda': 0.065,
                            'reg_alpha': 0.797,
                            'min_split_gain': 0.350,
                            'nthread': 8,
                            'seed': 42,
                            'scale_pos_weight':1.15,
                            'verbose': -1
        }
        
        model = lgb.train(base_params_lgb, train_set=train_matrix, num_boost_round=80000, valid_sets=valid_matrix, verbose_eval=-1, early_stopping_rounds=600)
        joblib.dump(model, '../model/lgb_cv_{}.pkl'.format(i))
        joblib.dump(target_enc, '../model/encoder_cv_{}.pkl'.format(i))

if __name__ == '__main__':
    train_path = '../data/train.csv'
    train_func(train_path)
