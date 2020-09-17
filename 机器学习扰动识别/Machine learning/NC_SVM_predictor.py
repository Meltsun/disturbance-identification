"""
临近种类支持向量机-预测器
参数已确定情况下，输出预测准确率
"""
import numpy as np               #基本计算
from Dataset_class import dataset #数据集类
from Dataset_class import preprocess
from sklearn.neighbors import KNeighborsClassifier #KN分类器
from sklearn.svm import SVC

#由基本的列表生成数据
totalData=preprocess()
totalData.minmax_standardize()
trainData,testData=totalData.split(0.8*750)

#总预测结果 None表示未完成分类
forecastResults=[None for i in range(0,testData.count_sample())]

#生成并训练10个SVM二分类器
svmClassifier={0: {1:SVC(kernel='rbf',gamma=1.6,C=1500), 
                   2:SVC(kernel='rbf',gamma=1.6,C=20), 
                   3:SVC(kernel='rbf',gamma=6,C=10), 
                   4:SVC(kernel='rbf',gamma=2,C=1)}, 
               1: {2:SVC(kernel='rbf',gamma=2,C=1000), 
                   3:SVC(kernel='rbf',gamma=0.7,C=5), 
                   4:SVC(kernel='rbf',gamma=4,C=4)},
               2: {3:SVC(kernel='rbf',gamma=4,C=100), 
                   4:SVC(kernel='rbf',gamma=0.8,C=150)}, 
               3: {4:SVC(kernel='rbf',gamma=0.5,C=1)}}
for i in range(0,4):
    for j in range(i+1,5):
        thisData=trainData.conditional_extract(i,j) 
        svmClassifier[i][j].fit(thisData.data,thisData.target)
thisData=None

#生成knn5分类器
knnClassifier=KNeighborsClassifier(n_neighbors=9,weights="distance",metric="manhattan")
knnClassifier.fit(trainData.data,trainData.target)

#得到验证样本的概率
knnResults=knnClassifier.predict_proba(testData.data)

#由knnResults分析得到 knn无法确定的样本的2个最大可能性
possibility={}
for i in range(0,testData.count_sample()):
    if(knnResults[i].max()==1):
        if(knnResults[i].argmax()==2):
            possibility[i]=[knnResults[i].argmax(),0]
        else:
            possibility[i]=[knnResults[i].argmax(),2]
    else:
        possibility[i]=[knnResults[i].argmax(),0]
        if(possibility[i][0]==0):
            possibility[i][-1]=1
        max=knnResults[i][possibility[i][-1]]
        for j in range(1,5):
            if(knnResults[i][j]>max and j!=possibility[i][0]):
                possibility[i][-1]=j
    possibility[i].sort()

#计算knn后的正确率
j=0
nRight=[0 for i in range(0,5)]
for i in range(0,testData.count_sample()):
    if(forecastResults[i]==None):
        if(testData.target[i] in possibility[i]):
            j+=1
            nRight[round(testData.target[i])]+=1
    else:
        if(testData.target[i]==forecastResults[i]):
            j+=1
            nRight[round(testData.target[i])]+=1

print(f"knn五选二正确率：{100*j/testData.count_sample():4f}%\n")

for i in range(0,5):
    print(f" {i} 正确率：{100*nRight[i]/testData.count_lable()[i]:4f}%")
print('')

##打印样本
#for i in range(0,testData.count_sample()):
#    if(forecastResults[i]==None):
#        print(possibility[i],end=',')
#    else:
#        print(forecastResults[i],end=',')
#    print(knnResults[i])

for i in possibility.keys():
    j=svmClassifier[possibility[i][0]][possibility[i][-1]].predict([testData.data[i]])[0]
    if(j==0):
        forecastResults[i]=possibility[i][0]
    elif(j==1):
        forecastResults[i]=possibility[i][-1]

j=0
nRight=[0 for i in range(0,5)]
for i in range(0,testData.count_sample()):
    if(testData.target[i]==forecastResults[i]):
        j+=1
        nRight[round(testData.target[i])]+=1

print(f"总正确率：{100*j/testData.count_sample():4f}%\n")

for i in range(0,5):
    print(f" {i} 正确率：{100*nRight[i]/testData.count_lable()[i]:4f}%")


        














   






