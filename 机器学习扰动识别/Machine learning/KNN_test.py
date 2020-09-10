#knn 测试
import numpy as np               #基本计算
from Dataset_class import dataset #数据集类
from Dataset_class import preprocess
from sklearn.neighbors import KNeighborsClassifier #KN分类器
import copy
#由基本的列表生成数据
totalData=preprocess()
totalData.minmax_standardize()

#生成数据集
trainData,testData=totalData.split(0.8*750)
#print(testData.count_lable())

#生成knn5分类器
knnClassifier=KNeighborsClassifier(n_neighbors=9,weights="distance",metric="manhattan")
knnClassifier.fit(trainData.data,trainData.target)

#输出5个概率
forecastResults=knnClassifier.predict_proba(testData.data)

#打印所有错误结果
statisticalResults={'1':[],'max and not 1':[],'second':[],'wrong and not 0':[],'0':[] }
for i in range(0,testData.count_sample()):
    if(forecastResults[i].argmax()+1==testData.target[i]):
        if(forecastResults[i][round(testData.target[i]-1)]==1):
            statisticalResults['1']+=[i]
        else:
            statisticalResults['max and not 1']+=[i]
    else:
        if(forecastResults[i][round(testData.target[i]-1)]==0):
            statisticalResults['0']+=[i]
        else:
            x=copy.deepcopy(forecastResults[i])
            x[x.argmax()]=0
            if(x.argmax()+1==testData.target[i]):
                statisticalResults['second']+=[i]
            else:
                statisticalResults['wrong and not 0']+=[i]

for i in statisticalResults.keys():
    print('\n',i)
    for j in statisticalResults[i]:
        print(forecastResults[j],testData.target[j])

for i in statisticalResults.keys():
    print(i,':',len(statisticalResults[i]))
