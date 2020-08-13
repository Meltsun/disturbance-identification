"""
main
临近种类支持向量机-交叉验证器
求超参数
说明：目前为学习版本的代码
"""
import numpy as np
from sklearn.model_selection import cross_val_score #交叉验证函数
from preprocesser import dataset #数据集类
from sklearn.svm import SVC      #SVM分类器
from preprocesser import preprocess

trainData=preprocess().conditional_extract(1,2)
trainData.minmax_standardize()
trainData.shuffle()
trainData.print()

iteratorC=iter([0.01,0.1,1,10,100,1000,5000,10000,15000,20000,25000,30000])
C=0
bestScore=0
bestC=0
bestGamma=0
k=max(trainData.nLable.values())

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
	print(f"当前最高准确率：{bestScore*100:.2f}% \n C：{C} \n gamma：{Gamma}")
print(f"最高准确率：{bestScore*100:.2f}%")

print(f"交叉验证成功 C:{svmC}, gamma:{svmGamma}")
		
