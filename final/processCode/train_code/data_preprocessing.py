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
import re
from urllib.parse import urlparse
from utils import *

class YGCheckICP(object):
    topRootDomain = (
        '.com', '.la', '.io', '.co', '.info', '.net', '.org', '.me', '.mobi',
        '.us', '.biz', '.xxx', '.ca', '.co.jp', '.com.cn', '.net.cn',
        '.org.cn', '.mx', '.tv', '.ws', '.ag', '.com.ag', '.net.ag',
        '.org.ag', '.am', '.asia', '.at', '.be', '.com.br', '.net.br',
        '.bz', '.com.bz', '.net.bz', '.cc', '.com.co', '.net.co',
        '.nom.co', '.de', '.es', '.com.es', '.nom.es', '.org.es',
        '.eu', '.fm', '.fr', '.gs', '.in', '.co.in', '.firm.in', '.gen.in',
        '.ind.in', '.net.in', '.org.in', '.it', '.jobs', '.jp', '.ms',
        '.com.mx', '.nl', '.nu', '.co.nz', '.net.nz', '.org.nz',
        '.se', '.tc', '.tk', '.tw', '.com.tw', '.idv.tw', '.org.tw',
        '.hk', '.co.uk', '.me.uk', '.org.uk', '.vg', ".com.hk")

    @classmethod
    def get_domain_root(cls, url):
        domain_root = ""
        ## 若不是 http或https开头，则补上方便正则匹配规则
        if url[0:4] != "http" and url[0:5] != "https":
            url = "http://" + url

        reg = r'[^\.]+(' + '|'.join([h.replace('.', r'\.') for h in YGCheckICP.topRootDomain]) + ')$'
        pattern = re.compile(reg, re.IGNORECASE)

        parts = urlparse(url)
        host = parts.netloc
        m = pattern.search(host)
        res = m.group() if m else host
        domain_root = "-" if not res else res
        return domain_root



def feature_engineering(data):

    data['country'],data['temp'] = data['tlsSubject'].str.extract('C=(.*?)([,/\s]|$)', expand=True)
    data['state'],data['temp'] = data['tlsSubject'].str.extract('ST=(.*?)([,/\s]|$)', expand=True)
    data['organization'],data['temp'] = data['tlsSubject'].str.extract('O=(.*?)([,/]|$)', expand=True) #持有组织
    data['OU'],data['temp'] = data['tlsSubject'].str.extract('OU=(.*?)([,/]|$)', expand=True)
    data['CN'],data['temp'] = data['tlsSubject'].str.extract('CN=(.*?)([,/]|$)', expand=True)
    
    data['issueCountry'],data['temp'] = data['tlsIssuerDn'].str.extract('C=(.*?)([,/\s]|$)', expand=True)
    data['issueState'],data['temp'] = data['tlsIssuerDn'].str.extract('ST=(.*?)([,/\s]|$)', expand=True)
    data['issueOrganization'],data['temp'] = data['tlsIssuerDn'].str.extract('O=(.*?)([,/]|$)', expand=True) #签发组织
    data['issueOU'],data['temp'] = data['tlsIssuerDn'].str.extract('OU=(.*?)([,]|$)', expand=True)
    data['issueCN'],data['temp'] = data['tlsIssuerDn'].str.extract('CN=(.*?)([,/]|$)', expand=True)
    #  这段代码写的很dirty，但是不执行上面一步 在这一步会有维度异常，原因未知

    data['country'] = data['tlsSubject'].str.extract('C=(.*?)([,/\s]|$)', expand=True)
    data['state'] = data['tlsSubject'].str.extract('ST=(.*?)([,/\s]|$)', expand=True)
    data['organization'] = data['tlsSubject'].str.extract('O=(.*?)([,/]|$)', expand=True) #持有组织
    data['OU'] = data['tlsSubject'].str.extract('OU=(.*?)([,/]|$)', expand=True)
    data['CN'] = data['tlsSubject'].str.extract('CN=(.*?)([,/]|$)', expand=True)
    
    data['issueCountry'] = data['tlsIssuerDn'].str.extract('C=(.*?)([,/\s]|$)', expand=True)
    data['issueState'] = data['tlsIssuerDn'].str.extract('ST=(.*?)([,/\s]|$)', expand=True)
    data['issueOrganization'] = data['tlsIssuerDn'].str.extract('O=(.*?)([,/]|$)', expand=True) #签发组织
    data['issueOU'] = data['tlsIssuerDn'].str.extract('OU=(.*?)([,/]|$)', expand=True)
    data['issueCN'] = data['tlsIssuerDn'].str.extract('CN=(.*?)([,/]|$)', expand=True)
    
    for item in ['北京市','北京','beijing','BeiJing','BJ','北京帿','北京眿']:
        data.loc[data['state']==item,'state'] = 'Beijing'
        data.loc[data['issueState']==item,'issueState'] = 'Beijing'
    for item in ['ZheJiang','浙江省','ZJ','浙江','zhejiang','HANGZHOU','zj','浙江帿','浙江眿']:
        data.loc[data['state']==item,'state'] = 'Zhejiang'
        data.loc[data['issueState']==item,'issueState'] = 'Zhejiang'
    for item in ['guangdong','Guangdong Province','广东省','广东','Guangdong Sheng','GuangDong','GD','广东帿','广东眿']:
        data.loc[data['state']==item,'state'] = 'Guangdong'
        data.loc[data['issueState']==item,'issueState'] = 'Guangdong'
    for item in ['上海市','上海','上海帿','上海眿']:
        data.loc[data['state']==item,'state'] = 'Shanghai'
        data.loc[data['issueState']==item,'issueState'] = 'Shanghai'
    for item in ['jiangsu','江苏省','江苏','JiangSu','JS','江苏帿','江苏眿']:
        data.loc[data['state']==item,'state'] = 'Jiangsu'
        data.loc[data['issueState']==item,'issueState'] = 'Jiangsu'
    for item in ['天津市','TJ','天津帿','天津眿']:
        data.loc[data['state']==item,'state'] = 'Tianjin'
        data.loc[data['issueState']==item,'issueState'] = 'Tianjin'
    for item in ['福建省','福建','福建帿','福建眿']:
        data.loc[data['state']==item,'state'] = 'Fujian'
        data.loc[data['issueState']==item,'issueState'] = 'Fujian'
    for item in ['河南','河南省','河南帿','河南眿']:
        data.loc[data['state']==item,'state'] = 'Henan'
        data.loc[data['issueState']==item,'issueState'] = 'Henan'
    for item in ['安徽','安徽省','安徽帿','安徽眿']:
        data.loc[data['state']==item,'state'] = 'Anhui'
        data.loc[data['issueState']==item,'issueState'] = 'Anhui'
    for item in ['山东','shandong','山东省','山东帿','山东眿']:
        data.loc[data['state']==item,'state'] = 'Shandong'
        data.loc[data['issueState']==item,'issueState'] = 'Shandong'
    for item in ['Xinjiang Uygur Autonomous Region','新疆','新疆帿','新疆眿']:
        data.loc[data['state']==item,'state'] = 'Xinjiang'
        data.loc[data['issueState']==item,'issueState'] = 'Xinjiang'
    for item in ['湖南省','湖南','hunan','HuNan','湖南帿','湖南眿']:
        data.loc[data['state']==item,'state'] = 'Hunan'
        data.loc[data['issueState']==item,'issueState'] = 'Hunan'
    for item in ['湖北省','湖北','hubei','HuBei','湖北帿','湖北眿']:
        data.loc[data['state']==item,'state'] = 'Hubei'
        data.loc[data['issueState']==item,'issueState'] = 'Hubei'
    for item in ['陕西省','陕西','shanxi','ShanXi','陕西帿','陕西眿']:
        data.loc[data['state']==item,'state'] = 'Shanxi'
        data.loc[data['issueState']==item,'issueState'] = 'Shanxi'
    for item in ['重庆','重庆市','chongqing','ChongQing','重庆帿','重庆眿']:
        data.loc[data['state']==item,'state'] = 'Chongqing'
        data.loc[data['issueState']==item,'issueState'] = 'Chongqing'
    for item in ['内蒙古','内蒙古自治区','neimenggu','内蒙古帿','内蒙古眿']:
        data.loc[data['state']==item,'state'] = 'Neimenggu'
        data.loc[data['issueState']==item,'issueState'] = 'Neimenggu'

    data['downUpRatio'] = data['bytesOut'] / data['bytesIn']
    data['PktSizeAvg'] = (data['bytesIn'] + data['bytesOut']) / (data['pktsIn']+data['pktsOut'])
    data['downPktSizeAvg'] = data['bytesOut'] / data['pktsIn']
    data['upPktSizeAvg'] = data['bytesIn'] / data['pktsOut']
    
    
    # --> Good Feature
    data['pktsDelta'] = data['pktsIn'] - data['pktsOut']
    data['bytesDelta'] = data['bytesIn'] - data['bytesOut']
    data['AvgSizeDelta'] = data['downPktSizeAvg'] - data['upPktSizeAvg']

    data.loc[data['issueOrganization']==data['organization'],'selfCA'] = 1.0
    data.loc[(data['issueOrganization']!=data['organization']),'selfCA'] = 0.0
    data.loc[(data['issueOrganization'].isnull())&(data['organization'].isnull()),'selfCA'] = float("NaN")
    
    data.loc[data['issueCN']==data['CN'],'selfCA'] = 1.0
    
    data['domain_length'] = data.tlsSni.str.len()
    data['tlsSubject_length'] = data.tlsSubject.str.len()
    data['tlsIssuer_length'] = data.tlsIssuerDn.str.len()
    
    data['tlsDelta'] = data['tlsSubject_length'] - data['tlsIssuer_length']
    data['CN_domain_length'] = data.CN.str.len()
    
    data["organization_is_DefaultProb"] = data.organization.str.endswith(("Default Company Ltd","Internet Widgits Pty Ltd","My Company Name LTD.","1","Global Security","XX"))
    data.organization_is_DefaultProb = data.organization_is_DefaultProb.apply(lambda x:1.0 if(x==True) else(0.0 if(x==False) else float("NaN")))

    data['confusion'] = data[data.tlsSni.notna()].tlsSni.apply(lambda x:(len(split(x))-1)/len(x))
    data['confusion_CN'] = data[(data.CN.notna())&(data.CN!='')].CN.apply(lambda x:(len(split(x))-1)/len(x))


    data['SubjectItemCount'] = data['tlsSubject'].str.count("=")
    data['IssuerItemCount'] = data['tlsIssuerDn'].str.count("=")

    data['SubjectIssuerDelta'] = data['IssuerItemCount'] - data['SubjectItemCount']
    data["srcPortFlag"] = data["srcPort"]<49152 #https://www.imperial.ac.uk/media/imperial-college/faculty-of-engineering/computing/public/1819-pg-projects/Detecting-Malware-in-TLS-Traf%EF%AC%81c.pdf
    data.srcPortFlag = data.srcPortFlag.apply(lambda x:1 if(x==True) else 0)

    data['tlsSni_pointCnt'] = data[data['tlsSni'].notna()].tlsSni.apply(lambda x: x.count('.'))
    data['tlsSni_point_rate'] = data[data['tlsSni'].notna()].tlsSni.apply(lambda x: x.count('.') / len(x))
    data['tlsSni_alp_rate'] = data[data['tlsSni'].notna()].tlsSni.apply(lambda x: len(re.findall('[a-zA-Z]', x)) / len(x))
    data['tlsSni_num_rate'] = data[data['tlsSni'].notna()].tlsSni.apply(lambda x: len(re.findall('[0-9]', x)) / len(x))

    data['CN_pointCnt'] = data[data['CN'].notna()].CN.apply(lambda x: x.count('.'))
    data['CN_point_rate'] = data[(data.CN.notna())&(data.CN!='')].CN.apply(lambda x: x.count('.') / len(x))
    data['CN_alp_rate'] = data[(data.CN.notna())&(data.CN!='')].CN.apply(lambda x: len(re.findall('[a-zA-Z]', x)) / len(x))
    data['CN_num_rate'] = data[(data.CN.notna())&(data.CN!='')].CN.apply(lambda x: len(re.findall('[0-9]', x)) / len(x))


    data.tlsSni = data[data.tlsSni.notna()].tlsSni.apply(lambda x: YGCheckICP.get_domain_root(x))
    data.CN = data[data.CN.notna()].CN.apply(lambda x: YGCheckICP.get_domain_root(x))
    data.loc[~data['tlsVersion'].isin(['TLS 1.2','TLS 1.3','TLSv1','UNDETERMINED']),'tlsVersion'] = 'OTHER'


    alexa = pd.read_csv("AlexaTop50000.csv")
    alexaList = list(alexa.domain)

    d = {}
    for i in alexaList:
        d[i] = True

    for i in range(data.shape[0]):
        flag_a = d.get(data.loc[i]['CN'])
        flag_b = d.get(data.loc[i]['tlsSni'])
        if flag_a==True or flag_b==True:
            data.loc[i,'IsAlexaTop10000'] = 1.0
        elif flag_a==False or flag_b==False:
            data.loc[i,'IsAlexaTop10000'] = 0.0
        else:
            data.loc[i,'IsAlexaTop10000'] = float("NaN")    

    data.loc[data.country.isin(['CNstore','cn','HK']),'country'] = 'CN'
    data.loc[(~data.country.isin(['CN','US','XX','AU','GB','--']))&(data['country'].notna()),'country'] = 'OTHER'
    
    data.loc[data.issueCountry.isin(['CNstore','cn','HK']),'issueCountry'] = 'CN'
    data.loc[(~data.issueCountry.isin(['CN','US','XX','AU','GB','--']))&(data['issueCountry'].notna()),'issueCountry'] = 'OTHER'
 
    data.loc[data.state=='WA','state'] = 'Washington'
    data.loc[data.state=='CA','state'] = 'California'
    data.loc[data.state=='SomeState','state'] = 'Some-State'
    
    data.loc[data.issueState=='WA','issueState'] = 'Washington'
    data.loc[data.issueState=='CA','issueState'] = 'California'
    data.loc[data.issueState=='SomeState','issueState'] = 'Some-State'
    
    data['tlsSni_alp_rate_tlsSni_point_rate'] = data['tlsSni_alp_rate'] * data['tlsSni_point_rate']#ok
    data['confusion_tlsSni_pointCnt'] = data['confusion'] * data['tlsSni_pointCnt']


    data.drop(['appProtocol','tlsSubject','tlsIssuerDn','temp'],inplace=True,axis=1)

    return data
