# 项目来源
[基于人工智能的药物分子筛选](http://www.dcjingsai.com/common/cmpt/%E5%9F%BA%E4%BA%8E%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E7%9A%84%E8%8D%AF%E7%89%A9%E5%88%86%E5%AD%90%E7%AD%9B%E9%80%89_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html?slxydc=f0d607)  
## 竞赛背景
晶泰科技(XtalPi)是一家世界领先的以计算驱动创新的药物研发科技公司，应用量子物理、分子动力学、人工智能与云计算等技术，为全球创新药企提供智能化药物研发科技，提高药物发现的效率与成功率，降低研发成本，为患者带来更多优质高效的药物。目前已与多家世界顶级药企与科研机构建立深度合作，晶泰科技还成功的将AI算法引用到药物发现的过程中，其开发的深度学习模型在药物发现和结构优化上都有优异的表现，并且其药物固相筛选与设计平台是目前行业最先进的解决方案。

蛋白质担负着生命体的各种生理功能，是生物性状的直接表达者。蛋白质与小分子化合物的相互作用是进行药物设计的基础。在分子水平上深入研究蛋白质与药物分子的结合机理有助于为筛选及研发药效高、应用广及毒副作用小的新药提供丰富的计算依据，大大缩短现有的实验发现流程并降低临床失败风险。利用人工智能构建蛋白质和小分子的亲和力预测模型，用于筛选有效的药物候选分子，将大大加快药物研发流程，让患者得到最及时的治疗。  
## 任务
2014年，一种未知的疾病在全球肆虐，让人类束手无策。致病蛋白质很多，它们的结构序列都藏在df_protein.csv 数据集中（Sequence特征）。经过科学家的不懈努力，能与这些致病蛋白相结合的小分子（df_molecule.csv中的Fingerprint特征表示了其结构）也被发现，并附上了它们的理化属性。此外，在df_affinity.csv数据集中，包含了蛋白质和小分子之间的亲和力数值（Ki特征）。 时间紧迫！作为算法科学家，你能够仅仅在六周时间里，从测试集（df_affinity_test_toBePredicted.csv）中预测出致病蛋白与小分子的亲和力值，从而找出最有效的药物分子吗？  
## 总体概述
主办方晶泰科技搜集到了2万条数据（部分原始数据来源于BindingDB数据库），包含了蛋白质与小分子亲和力预测值以及蛋白质的一级结构序列，还有小分子的分子指纹及对应的18种物化属性。
根据蛋白质的信息，数据被分为两部分，一部分蛋白质的相关信息作为训练集，另一部分蛋白质的相关信息作为测试集，分别标注以train和test。         
## 数据文件解释  
### 1.蛋白质信息 df_protein.csv

数据共两列，分别是蛋白质id和蛋白质的一级结构序列的矢量化结果。关于结构序列是什么形式，可参考[wiki链接](https://zh.wikipedia.org/wiki/%E8%9B%8B%E7%99%BD%E8%B3%AA%E4%B8%80%E7%B4%9A%E7%B5%90%E6%A7%8B)

例如：  
|Protein_ID|Sequence |  
|---|---|  
|4|MEPVPSARAELQFSLLANVSDTFPSAFPSASANASGSPGARSASSLALAIAITALYSAVCAVGLLGNVLVMFGIVRYTKLKTATNIYIFNLALADALATSTLPFQSAKYLMETWPFGELLCKAVLSIDYYNMFTSIFTLTMMSVDRYIAVCHPVKALDFRTPAKAKLINICIWVLASGVGVPIMVMAVTQPRDGAVVCTLQFPSPSWYWDTVTKICVFLFAFVVPILIITVCYGLMLLRLRSVRLLSGSKEKD|  


### 2.蛋白质小分子亲和力值信息 df_affinity.csv 

数据共三列， 分别是蛋白质id， 小分子id，亲和力值Ki（是经过函数变换后的值）
例如
             
|Protein_ID|Molecule_ID|Ki|  
|---|---|---|  
|0|0|8.309803963|  
|1|1|10.29242992|

### 3.小分子信息 df_molecule.csv（会存在缺失值） 

数据共二十列，其中每个特征的含义如下：  
Molecule_ID：小分子id，  
Fingerprint：分子指纹；  
cyp_sc9, cyp_sa4, cyp_2d6 ：三种cyp酶；  
Ames_toxicity: ames毒性测试  
Fathead_minnow_toxicity：黑头呆鱼毒性测试  
Tetrahymena_pyriformis_toxicity：梨型四膜虫毒性测试  
Honey_bee_toxicity：蜜蜂毒性测试  
ell_permeability： 细胞渗透性  
LogP：油水分配数  
Renal_organic_cation_transporter： 肾脏阳离子运输性  
CLtotal： 血浆清除率  
Hia: 人体肠道吸收水平  
Biodegradation：生物降解水平  
Vdd：表观分布容积  
P_glycoprotein_inhibition： p糖蛋白抑制物  
NOAEL：无可见有害作用剂量  
Solubility：药物溶解度  
Bbb：血脑屏障  
Half-life：药物半衰期