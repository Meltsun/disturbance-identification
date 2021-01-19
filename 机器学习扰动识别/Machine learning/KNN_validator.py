"""
最临近-交叉验证器（旧）
求超参数
"""
import numpy as np
from sklearn.model_selection import cross_val_score #交叉验证 函数
from Dataset_class import dataset #数据集 类
from Dataset_class import obj_print#实例打印 函数
from Dataset_class import preprocess#生成数据集 函数
from sklearn import neighbors  #KN分类器
from Fisher_Select import Fisher_Select

trainData=preprocess()
ii=Fisher_Select(trainData.data,trainData.target)
trainData.minmax_standardize()
trainData.shuffle()
obj_print(trainData)

iteratorC=iter([0.01,0.1,1,10,100,1000,5000,10000,15000,20000,25000,30000])
C=0
bestScore=0
bestC=0
bestGamma=0
k=max(trainData.count_lable().values)

while(C<30000):
	C=next(iteratorC)
	iteratorGamma=iter([0.0001,0.001,0.01,0.1,0.5,1,2,3,4,5,6,7,8])
	Gamma=0
	while(Gamma<8):
		Gamma=next(iteratorGamma)
		Classifier=SVC(kernel='rbf',gamma=Gamma,C=C)
		Score=cross_val_score(Classifier, trainData.data, trainData.target, cv=k,scoring='accuracy').mean()
		if(Score>bestScore):
			bestScore=Score
			bestC=C
			bestGamma=Gamma
		print(f"C：{C} \n gamma：{Gamma}\n准确率：{Score*100:.2f}% \n ")
print(f"最高准确率：{bestScore*100:.2f}%")
print(f"交叉验证成功 C:{bestC}, gamma:{bestGamma}")
		


