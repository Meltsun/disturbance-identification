"""
main
广义神经网络-预测器
参数已确定情况下，输出预测准确率
"""


import numpy as np               #基本计算
from Dataset_class import dataset #数据集类
from sklearn import tree
from Dataset_class import preprocess
import graphviz

#由基本的列表生成数据
totalData=preprocess()
totalData.minmax_standardize()
trainData,testData=totalData.split(0.8*750)

#训练
Classifier=tree.DecisionTreeClassifier()
Classifier.fit(trainData.data,trainData.target)

dot_data = tree.export_graphviz(Classifier, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

#验证，输出正确率
forecastResults=Classifier.predict(testData.data)
nRight=0
for i in range(0,len(testData.target)):
	if(testData.target[i]==forecastResults[i]):
		nRight+=1
print(f"正确率：{100*nRight/len(testData.target):2f}%")

