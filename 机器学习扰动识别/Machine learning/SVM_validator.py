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

class iterator:#迭代器
	c=0
	min=0
	max=0
	a=0
	def __init__(self,min,max):
		self.min=min;
		self.a=min
		self.max=max;
		return None

	def __iter__(self):
		self.c=0
		return self

	def __next__(self):
		if(self.c>self.max):
			return self.max
		self.c=self.c+self.a
		if(self.c>=self.a*10 and self.a<self.min*100):
			self.a*=10
		return  self.c
	
def cross_validator(trainData):#遍历迭代器中所有超参数
	iteratorC=iter(iterator(1,32000))
	C=0
	bestScore=0
	bestC=0
	bestGamma=0
	while(C<32000):
		C=next(iteratorC)
		iteratorGamma=iter(iterator(0.01,10))
		Gamma=0
		print(f"当前进度：C-{C/320:.4f}% ")
		while(Gamma<1):
			Gamma=next(iteratorGamma)
			Classifier=SVC(kernel='rbf',gamma=Gamma,C=C)
			Score=cross_val_score(Classifier, trainData.data, trainData.target, cv=5,scoring='accuracy').mean()
			if(Score>bestScore):
				bestScore=Score
				bestC=C
				bestGamma=Gamma
		print(f"当前最高准确率：{bestScore*100:.2f}% \n C：{C} \n gamma：{Gamma}")
	print(f"最高准确率：{bestScore*100:.2f}%")
	return bestC,bestGamma


#导入数据
file=open('D:\\大创\\兵王.txt')
file1=file.readlines()
file.close()

#由基本的列表生成数据
totalData=dataset().build_from_outside(file1)
#随机分割出训练集
trainData,testData=totalData.split(0.2)

svmC,svmGamma=cross_validator(trainData)

print(f"交叉验证成功 C:{svmC}, gamma:{svmGamma}")
		
