###################################start###################################
##加载包
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
feat = feat.astype('int')

feat.columns=["Fingerprint_{0}".format(i) for i in range(0,167)]
feat["Molecule_ID"] = df_molecule['Molecule_ID']#111216,168
data = data.merge(feat, on='Molecule_ID', how='left')#206467,170

#2、df_molecule其他特征处理
feat = df_molecule.drop('Fingerprint',axis=1)#111216,19
data = data.merge(feat, on='Molecule_ID', how='left')#206467,188


#3、protein 蛋白质 词向量训练
n = 250
texts = [[word for word in re.findall(r'.{1}',document)] 
               for document in list(protein_concat['Sequence'])]

model = Word2Vec(texts,size=n,window=10,min_count=1,negative=3,
                 sg=1,sample=0.001,hs=1,workers=4)  #window=4,min_count=1,negative=3,sg=1,sample=0.001,hs=1,workers=4
 

vectors = pd.DataFrame([model[word] for word in (model.wv.vocab)])
vectors['Word'] = list(model.wv.vocab)
vectors.columns= ["vec_{0}".format(i) for i in range(0,n)]+["Word"]

wide_vec = pd.DataFrame()
result1=[]
aa = list(protein_concat['Protein_ID'])
for i in range(len(texts)):
    result2=[]         
    for w in range(len(texts[i])):
        result2.append(aa[i])    
    result1.extend(result2)
wide_vec['Id'] = result1

result1=[]
for i in range(len(texts)):
    result2=[]         
    for w in range(len(texts[i])):
        result2.append(texts[i][w])    
    result1.extend(result2)
wide_vec['Word'] = result1

del result1,result2

wide_vec = wide_vec.merge(vectors,on='Word', how='left')
wide_vec = wide_vec.drop('Word',axis=1)
wide_vec.columns = ['Protein_ID']+["vec_{0}".format(i) for i in range(0,n)]

del vectors

name = ["vec_{0}".format(i) for i in range(0,n)]

feat = pd.DataFrame(wide_vec.groupby(['Protein_ID'])[name].agg('mean')).reset_index()
feat.columns=["Protein_ID"]+["mean_ci_{0}".format(i) for i in range(0,n)]
data = data.merge(feat, on='Protein_ID', how='left')

#################################### lgb ############################

train_feat = data[data['Ki']> -11].fillna(0)
testt_feat = data[data['Ki']<=-11].fillna(0)
label_x  = train_feat['Ki']
label_y  = testt_feat['Ki']

submission = testt_feat[['Protein_ID','Molecule_ID']]
len(testt_feat)
train_feat = train_feat.drop('Ki',axis=1)
testt_feat = testt_feat.drop('Ki',axis=1)
train_feat = train_feat.drop('Protein_ID',axis=1)
testt_feat = testt_feat.drop('Protein_ID',axis=1)
train_feat = train_feat.drop('Molecule_ID',axis=1)
testt_feat = testt_feat.drop('Molecule_ID',axis=1)


#lgb算法
train = lgb.Dataset(train_feat, label=label_x)
test  = lgb.Dataset(testt_feat, label=label_y,reference=train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': 'l2',
    #'objective': 'multiclass',
    #'metric': 'multi_error',
    #'num_class':5,
	#'min_data_in_leaf':1000,
    'min_child_weight': 10,#3
	#'max_depth':7,#5
    'num_leaves': 80,#2 ** 5
    'lambda_l2': 10,#10
    'subsample': 0.7,#0.7
    'colsample_bytree': 0.7,#0.7
    'colsample_bylevel': 0.7,#0.7
    'learning_rate': 0.1,#0.05
    'tree_method': 'exact',
    'seed': 2017,#2017
    'nthread': 12,#12
    'silent': True
    }

num_round = 5000#3000
gbm = lgb.train(params, 
                  train, 
                  num_round, 
                  verbose_eval=50,
                  valid_sets=[train,test]
                  )

preds_sub = gbm.predict(testt_feat)


#结果保存
nowTime=datetime.datetime.now().strftime('%m%d%H%M')#现在
name='mkd_'+nowTime+'.csv'
submission['Ki'] = preds_sub
submission.to_csv(name, index=False)