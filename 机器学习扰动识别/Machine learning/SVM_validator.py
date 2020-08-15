"""
main
支持向量机-交叉验证器
求超参数
"""
import numpy as np
from sklearn.model_selection import cross_val_score #交叉验证 函数
from preprocesser import dataset #数据集 类
from preprocesser import obj_print#实例打印 函数
from preprocesser import preprocess#生成数据集 函数
from sklearn.svm import SVC      #SVM分类器 类

totalData=preprocess()
result=[ [{'score':0,'C':0,'gamma':0} for i in range(0,6)] for i in range(0,6)]



for i in range(1,6):
	for j in range(i+1,6):
		trainData=totalData.conditional_extract(i,j)
		trainData.minmax_standardize()
		trainData.shuffle()
		print("■■■■■■■■■■■■■■■■■■■■■■■■■■■")
		obj_print(trainData)

		iteratorC=iter([0.01,0.1,1,10,100,1000,5000,10000,15000,20000,25000,30000])
		C=0
		k=max(trainData.count_lable().values())

		while(C<30000):
			C=next(iteratorC)
			iteratorGamma=iter([0.0001,0.001,0.01,0.1,0.5,1,2,3,4,5,6,7,8])
			Gamma=0
			while(Gamma<8):
				Gamma=next(iteratorGamma)
				Classifier=SVC(kernel='rbf',gamma=Gamma,C=C)
				Score=cross_val_score(Classifier, trainData.data, trainData.target, cv=k,scoring='accuracy').mean()
				if(Score>result[i][j]['score']):
					result[i][j]['score']=Score
					result[i][j]['c']=C
					result[i][j]['gamma']=Gamma
			print(f"{i} vs {j}  ,C：{C} ,best score:{result[i][j]['score']*100:.2f}")


print("■■■■■■■■■■■■■■■■■■■■■■■■■■■")
for i in range(1,6):
	for j in range(i+1,6):
		print(f"\n{i} vs {j}")
		print(result[i][j])

