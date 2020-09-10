"""
main
支持向量机-交叉验证器
求超参数
"""
import numpy as np
from sklearn.model_selection import cross_val_score #交叉验证 函数
from Dataset_class import dataset #数据集 类
from Dataset_class import obj_print#实例打印 函数
from Dataset_class import preprocess#生成数据集 函数
from sklearn.svm import SVC      #SVM分类器 类

totalData=preprocess()
totalData.shuffle()
totalData.minmax_standardize()
trainData,testData=totalData.split(750*0.8)
trainData=trainData.conditional_extract([4],[2,1,3,0])
testData=testData.conditional_extract([4],[2,1,3,0])

print("■■■■■■■■■■■■■■■■■■■■■■■■■■■")
obj_print(trainData)

Classifier=SVC(kernel='rbf')
Classifier.fit(trainData.data,trainData.target)
result=Classifier.predict(testData.data)		

print("■■■■■■■■■■■■■■■■■■■■■■■■■■■")

j=0
for i in range(0,testData.count_sample()):
	if(result[i]==testData.target[i]):
		j+=1
print(j)
print(testData.count_sample())

