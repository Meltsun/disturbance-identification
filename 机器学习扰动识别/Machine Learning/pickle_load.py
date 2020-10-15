"""
临近种类支持向量机-预测器
参数已确定情况下，输出预测准确率
"""
import numpy as np               #基本计算
from Dataset_class import dataset #数据集类
from Dataset_class import preprocess
from sklearn.neighbors import KNeighborsClassifier #KN分类器
from sklearn.svm import SVC
import pickle

#由基本的列表生成数据
totalData=preprocess()
totalData.minmax_standardize()

#总预测结果 None表示未完成分类
forecastResults=[None for i in range(0,totalData.count_sample())]

#载入10个SVM二分类器
file=open('svmClassifiers.pkl','rb')
svmClassifier=pickle.load(file)
file.close()

#载入knn5分类器
file=open('knnClassifier.pkl','rb')
knnClassifier=pickle.load(file)
file.close()

#得到验证样本的概率
knnResults=knnClassifier.predict_proba(totalData.data)

#由knnResults分析得到 knn无法确定的样本的2个最大可能性
possibility={}
for i in range(0,totalData.count_sample()):
    if(knnResults[i].max()==1):
        if(knnResults[i].argmax()==4):
            possibility[i]=[knnResults[i].argmax(),2]
        else:
            possibility[i]=[knnResults[i].argmax(),4]
    else:
        possibility[i]=[knnResults[i].argmax(),0]
        if(possibility[i][0]==0):
            possibility[i][-1]=1
        max=knnResults[i][possibility[i][-1]]
        for j in range(0,5):
            if(knnResults[i][j]>max and j not in possibility[i]):
                possibility[i][-1]=j
    possibility[i].sort()

#计算knn后的正确率
j=0
nRight=[0 for i in range(0,5)]
for i in range(0,totalData.count_sample()):
    if(forecastResults[i]==None):
        if(totalData.target[i] in possibility[i]):
            j+=1
            nRight[round(totalData.target[i])]+=1
    else:
        if(totalData.target[i]==forecastResults[i]):
            j+=1
            nRight[round(totalData.target[i])]+=1

print(f"knn五选二正确率：{100*j/totalData.count_sample():4f}%\n")


for i in range(0,5):
    print(f" {i} 正确率：{100*nRight[i]/totalData.count_lable()[i]:4f}%")
print('')

##打印样本
#for i in range(0,totalData.count_sample()):
#    if(forecastResults[i]==None):
#        print(possibility[i],end=',')
#    else:
#        print(forecastResults[i],end=',')
#    print(knnResults[i])

for i in possibility.keys():
    j=svmClassifier[possibility[i][0]][possibility[i][-1]].predict([totalData.data[i]])[0]
    if(j==0):
        forecastResults[i]=possibility[i][0]
    elif(j==1):
        forecastResults[i]=possibility[i][-1]

j=0
nRight=[0 for i in range(0,5)]
for i in range(0,totalData.count_sample()):
    if(totalData.target[i]==forecastResults[i]):
        j+=1
        nRight[round(totalData.target[i])]+=1

print(f"总正确率：{100*j/totalData.count_sample():4f}%\n")

for i in range(0,5):
    print(f" {i} 正确率：{100*nRight[i]/totalData.count_lable()[i]:4f}%")


        














   







