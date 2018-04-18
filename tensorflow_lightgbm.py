# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 08:52:13 2018

@author: yousa
"""

import gc
import re
import sys
import time
import jieba
import os.path
import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
import gensim 
from gensim.models import Word2Vec  

################################# data read ################################
#工作空间设置
data_path = 'E:/work/protein/'
os.chdir(data_path)#设置当前工作空间
print (os.getcwd())#获得当前工作目录

#数据读取
df_protein_train    = pd.read_csv('df_protein_train.csv')              #1653,2
df_protein_test     = pd.read_csv('df_protein_test.csv')               #414,2
protein_concat      = pd.concat([df_protein_train,df_protein_test])    #2067,2
df_molecule         = pd.read_csv('df_molecule.csv')                   #111216,20
df_affinity_train   = pd.read_csv('df_affinity_train.csv')             #165084,3
df_affinity_test    = pd.read_csv('df_affinity_test_toBePredicted.csv')#41383,2
df_affinity_test['Ki'] = -11                                           #41383,3
data  =  pd.concat([df_affinity_train,df_affinity_test])               #206467,3
  
##############################################################################
###########                            feature                     ###########
##############################################################################
#1、Fingerprint分子指纹处理展开
feat = []
for i in range(0,len(df_molecule)):
    feat.append(df_molecule['Fingerprint'][i].split(','))
feat = pd.DataFrame(feat)
feat = feat.astype('int')#111216,167

feat.columns=["Fingerprint_{0}".format(i) for i in range(0,167)]
feat["Molecule_ID"] = df_molecule['Molecule_ID']#111216,168
data = data.merge(feat, on='Molecule_ID', how='left')#206467,170

#2、df_molecule其他特征处理
feat = df_molecule.drop('Fingerprint',axis=1)#111216,19
data = data.merge(feat, on='Molecule_ID', how='left')#206467,188


#3、protein 蛋白质 词向量训练
n = 128
texts = [[word for word in re.findall(r'.{3}',document)] 
               for document in list(protein_concat['Sequence'])]

