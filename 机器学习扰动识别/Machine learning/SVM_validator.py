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

listC={1000:[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,0],
   10:[5,10,15,20,25,30,35,40,45,50,0],
   1:[0.5,1,2,3,4,5,0],
   100:[50,100,150,200,250,300,350,400,450,500,0],
   }

listGamma={0.5:[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0],
		   1:[0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,0],
		   2:[1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,0],
		   4:[3.5,3.6,3.7,3.8,3.9,4,4.1,4.2,4.3,4.4,4.5,0],
		   6:[5.5,5.6,5.7,5.8,5.9,6,6.1,6.2,6.3,6.4,6.5,0]}

lastResult={(1,2):(2,1000),
			(1,3):(2,10),
			(1,4):(6,10),
			(1,5):(2,1),
			(2,3):(2,1000),
			(2,4):(1,10),
			(2,5):(4,1),
			(3,4):(4,100),
			(3,5):(0.5,100),
			(4,5):(0.5,1)}

for i in [2]:
	for j in [5]:
		trainData=totalData.conditional_extract(i,j)
		trainData.minmax_standardize()
		trainData.shuffle()
		print("■■■■■■■■■■■■■■■■■■■■■■■■■■■")
		obj_print(trainData)

		iteratorC=iter(listC[lastResult[(i,j)][1]])
		C=next(iteratorC)
		k=max(trainData.count_lable().values())

		while(C!=0):
			iteratorGamma=iter(listGamma[lastResult[(i,j)][0]])
			Gamma=next(iteratorGamma)
			while(Gamma!=0):
				Classifier=SVC(kernel='rbf',gamma=Gamma,C=C)
				Score=cross_val_score(Classifier, trainData.data, trainData.target, cv=k,scoring='accuracy').mean()
				if(Score>result[i][j]['score']):
					result[i][j]['score']=round(Score,8)
					result[i][j]['C']=C
					result[i][j]['gamma']=Gamma
				Gamma=next(iteratorGamma)
			print(f"{i} vs {j}  ,C：{C} ,best score:{result[i][j]['score']*100}")
			C=next(iteratorC)
			


print("■■■■■■■■■■■■■■■■■■■■■■■■■■■")
for i in range(1,6):
	for j in range(i+1,6):
		print(f"\n{i} vs {j}")
		print(result[i][j])

