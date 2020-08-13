"""
临近种类支持向量机-预测器
参数已确定情况下，输出预测准确率
"""
import numpy as np               #基本计算
from preprocesser import dataset #数据集类

#由基本的列表生成数据
totalData=data_preprocess()

#训练
trainData,testData=totalData.split(0.2)
Classifier=None

#验证，输出正确率
forecastResults=Classifier.predict(testData.data)
nRight=0
for i in range(0,len(testData.target)):
    if(testData.target[i]==forecastResults[i]):
        nRight+=1
print(f"正确率：{100*nRight/len(testData.target):2f}%")
        














   





