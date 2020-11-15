"""
临近种类支持向量机-预测器
参数已确定情况下，输出预测准确率
"""
import numpy as np               #基本计算
import pandas as pd 
from Dataset_class import dataset #数据集类
from Dataset_class import preprocess
from Dataset_class import obj_print
from sklearn.neighbors import KNeighborsClassifier #KN分类器
from sklearn.svm import SVC
import pickle
from sklearn import preprocessing

testID=pd.read_csv("testID.csv",header=None,index_col=None ).values.ravel()
trainID=[i for i in range(0,750) if i not in testID] 

#制作归一化器
trainData=preprocess()
standardizer = preprocessing.MinMaxScaler().fit(trainData.data)
file=open("standardizer.pkl",'wb')
pickle.dump(standardizer,file,4)

trainData.data = standardizer.transform(trainData.data[trainID])
trainData.target = trainData.target[trainID]

#print(trainData.count_lable())

#生成并训练10个SVM二分类器
svmClassifier={0: {1:SVC(kernel='rbf',gamma=2,C=0.5), 
                   2:SVC(kernel='rbf',gamma=0.8,C=5), 
                   3:SVC(kernel='rbf',gamma=1,C=8), 
                   4:SVC(kernel='rbf',gamma=1.8,C=10)}, 
               1: {2:SVC(kernel='rbf',gamma=2.225,C=9), 
                   3:SVC(kernel='rbf',gamma=2.4,C=5000), 
                   4:SVC(kernel='rbf',gamma=0.7,C=2)},
               2: {3:SVC(kernel='rbf',gamma=2.685,C=1.6), 
                   4:SVC(kernel='rbf',gamma=2.6,C=5)}, 
               3: {4:SVC(kernel='rbf',gamma=0.9,C=15)}}

for i in range(0,4):
    for j in range(i+1,5):
        thisData=trainData.conditional_extract(i,j) 
        svmClassifier[i][j].fit(thisData.data,thisData.target)
thisData=None

file=open("svmClassifiers.pkl",'wb')
pickle.dump(svmClassifier,file,4)
file.close()

#生成knn5分类器
knnClassifier=KNeighborsClassifier(n_neighbors=5,weights="distance",metric="manhattan")
knnClassifier.fit(trainData.data,trainData.target)
file=open("knnClassifier.pkl",'wb')
pickle.dump(knnClassifier,file,4)
file.close()


    
        














   






