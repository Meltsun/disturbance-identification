"""
临近种类支持向量机-预测器
参数已确定情况下，输出预测准确率
"""
import numpy as np               #基本计算
from preprocesser import dataset #数据集类
from preprocesser import preprocess
from sklearn.neighbors import KNeighborsClassifier #KN分类器

#由基本的列表生成数据
totalData=preprocess()
totalData.minmax_standardize()
trainData,testData=totalData.split(round(0.8*750))

#生成10个SVM二分类器
svmClassifier={1: {2:SVC(kernel='rbf',gamma=1.6,C=1500), 
                   3:SVC(kernel='rbf',gamma=1.6,C=20), 
                   4:SVC(kernel='rbf',gamma=6,C=10), 
                   5:SVC(kernel='rbf',gamma=2,C=1)}, 
               2: {3:SVC(kernel='rbf',gamma=2,C=1000), 
                   4:SVC(kernel='rbf',gamma=0.7,C=5), 
                   5:SVC(kernel='rbf',gamma=4,C=4)},
               3: {4:SVC(kernel='rbf',gamma=4,C=100), 
                   5:SVC(kernel='rbf',gamma=0.8,C=150)}, 
               4: {5:SVC(kernel='rbf',gamma=0.5,C=1)}}
for i in range(1,5):
    for j in range(i+1,6):
        thisData=trainData.conditional_extract(i,j) 
        svmClassifier[i][j].fit(thisData.data,thisData.target)
thisData=None

#生成knn5分类器
knnClassifier=KNeighborsClassifier(n_neighbors=5,weights="uniform",p=1)
knnClassifier.fit(trainData)

#输出5个概率
forecastResults=knnClassifier.predict_proba(testData.data)

nRight=0
for i in range(0,len(testData.target)):
    if(testData.target[i]==forecastResults[i]):
        nRight+=1
print(f"正确率：{100*nRight/len(testData.target):2f}%")
        














   






