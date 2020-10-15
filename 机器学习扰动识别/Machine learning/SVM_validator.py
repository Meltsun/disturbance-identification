"""
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
result=[ [{'score':0,'C':0,'gamma':0} for i in range(0,6)] for i in range(0,6)]

listC={1000:[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,0],
   10:[5,10,15,20,25,30,35,40,45,50,0],
   1:[0.5,1,2,3,4,5,0]
   }

listGamma={1:[0.5, 0.6 ,0.7 ,0.8 ,0.9 ,1  ,1.1 ,1.2 ,1.3 ,1.4 ,1.5 ,0],
		   2:[1.5, 1.6, 1.7, 1.8, 1.9, 2  ,2.1 ,2.2 ,2.3 ,2.4 ,2.5 ,0],
		   3:[2.5, 2.6, 2.7, 2.8, 2.9, 3,  3.1 ,3.2 ,3.3 ,3.4 ,3.5 ,3.8 ,0],
		   10:[3,5,6,7,8,9,10,15,0]} 

lastResult={(0,1):(1,2), 
			(0,2):(10,1),
			(0,3):(1,10),
			(0,4):(10,2),
			(1,2):(10,2),
			(1,3):(1000,2),
			(1,4):(1,1),
			(2,3):(1,3),
			(2,4):(10,3),
			(3,4):(10,1)}



for i in [3]:
	for j in [4]:
		trainData=totalData.conditional_extract(i,j)
		trainData.minmax_standardize()
		trainData.shuffle()
		print("■■■■■■■■■■■■■■■■■■■■■■■■■■■")
		#打印这个数据集
		#obj_print(trainData)

		iteratorC=iter([10,11,12,13,14,15,16,17,18,19,20,0])
		C=next(iteratorC)
		k=max(trainData.count_lable().values())

		while(C!=0):
			iteratorGamma=iter([0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1,0])
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
for i in range(0,4):
	for j in range(i+1,5):
		print(f"\n{i} vs {j}")
		print(result[i][j])

