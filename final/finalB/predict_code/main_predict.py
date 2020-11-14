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

def test_func(test_path,save_path):
    # 请填写测试代码
    test = pd.read_csv(test_path)
    # 选手不得改变格式，测试代码跑不通分数以零算
    # #####选手填写测试集处理逻辑,在指定文件夹下生成可提交的csv文件

    test_features_selected = feature_engineering(test.copy())
    origin_test = test_features_selected.copy()
    test_features_selected.drop(['appProtocol','tlsSubject','tlsIssuerDn','tlsSubject_FakeDomain','tlsSni_fromTor','temp','eventId'],inplace=True,axis=1)


    _pred10 = np.zeros((test.shape[0],))

    cat_features = ['tlsSni', 'tlsVersion', 'country','state','organization','OU','CN','issueCountry','issueState','issueOrganization','issueOU','issueCN','srcAddress','destAddress']
    
    test_features_selected_copy = test_features_selected.copy()

    for i in range(5):
        model = joblib.load('./lgb_cv_{}.pkl'.format(i))
        target_enc = joblib.load('./encoder_cv_{}.pkl'.format(i))
        # 对高维类别变量进行编码
        test_features_selected = test_features_selected_copy.copy()
        test_features_selected = test_features_selected.join(target_enc.transform(test_features_selected[cat_features]).add_suffix('_target'))
        test_features_selected.drop(cat_features,inplace=True,axis=1)
        _pred10 += model.predict(test_features_selected)

    bagging = _pred10 / 5

    for i in range(len(bagging)):
        if(bagging[i]>0.5):#0.464
            bagging[i]=1.0
        else:
            bagging[i]=0.0


    pred_df = pd.Series(bagging, name='label', index=test.eventId).reset_index()


    # 模型后处理, 拦截来自暗网的流量
    pred_white = origin_test[origin_test['eventId'].isin(pred_df[pred_df['label']==0]['eventId'].values)]


    for idx, row_data in pred_white.T.iteritems():
        fromTorFlag = row_data['tlsSni_fromTor']
        eventId = row_data['eventId']
        tlsFakeDomain = row_data['tlsSubject_FakeDomain']
        if(fromTorFlag==True):#暗网流量
            pred_df.loc[pred_df['eventId']==eventId,'label'] = 1.0
        if(tlsFakeDomain==True):#虚假域名
            pred_df.loc[pred_df['eventId']==eventId,'label'] = 1.0

    pred_df.to_csv(save_path + 'result.csv',index = False,encoding='utf-8')


if __name__ == '__main__':
    test_path = '../data/test_1.csv'
    sava_path = '../result/'
    test_func(test_path,sava_path)
