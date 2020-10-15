"""
数据集类

"""

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle

class dataset:
	nFeatures=40 #特征值数量。需要人工设置
	target=None
	data=None

	#构造方法
	def __init__(self,N=0):
		self.target=np.empty(N)
		self.data=np.empty([N,self.nFeatures])
		return None

	#计算样本数
	def count_sample(self):
		return len(self.target)

	#计算各个标签的样本数
	def count_lable(self):
		lables={}
		for i in range(0,self.count_sample()):
			thisTarget=self.target[i]
			if thisTarget in lables.keys(): 
				lables[thisTarget]+=1
			else:
				lables[thisTarget]=1
		return lables

	#从字符串列表生成数据集,待修改
	def build_from_file(self,file):
		for i in range(0,self.count_sample()):
			thisData=[eval(t) for t in file[i].split(",")]
			self.target[i]=thisData[-1]
			self.data[i]=thisData[:-1]
		return None

	#按照特定方式标准化
	def zscore_standardize(self):
		self.data=preprocessing.StandardScaler().fit_transform(self.data)
		return None

	def minmax_standardize(self):
		self.data=preprocessing.MinMaxScaler().fit_transform(self.data)
		return None

	#切分数据,输入为前一返回数据集的样本数量
	def split(self,N,randomState=None): 
		N=round(N)
		dataset1=dataset(self.count_sample()-N)
		dataset2=dataset(N)
		dataset1.data,dataset2.data,dataset1.target,dataset2.target=train_test_split(self.data,self.target, test_size=N,random_state=randomState,stratify=self.target)
		return dataset2,dataset1

	#根据条件生成数据集,输入为标签为所有标签为0的动作和所有为1的，返回生成的数据集
	def conditional_extract(self,condition0,condition1): 
		if(type(condition0)==type(0)):
			condition0=[condition0]
		if(type(condition1)==type(1)):
			condition1=[condition1]
		N=0
		lables=self.count_lable()
		for i in condition0+condition1:
			N+=lables[i]
		thisDataset=dataset(N)
		N=0
		for i in range(0,self.count_sample()):
			thisLable=self.target[i]
			if(thisLable in condition0):
				thisDataset.data[N]=self.data[i]
				thisDataset.target[N]=0
				N+=1
			elif(thisLable in condition1):
				thisDataset.data[N]=self.data[i]
				thisDataset.target[N]=1
				N+=1
		return thisDataset

	#打乱
	def shuffle(self,x=None):
		self.data,self.target=shuffle(self.data,self.target,random_state=x)
		return None

#数据导入和预处理，返回一个数据集
def preprocess(): 
	file=open("feature.csv")
	file1=file.readlines()
	file.close()
	totalData=dataset(len(file1))
	totalData.build_from_file(file1)
	print(f"\n各类样本数量:")
	print(totalData.count_lable())
	print('')
	return totalData

#打印实例的所有属性
def obj_print(self): 
	print('\n'.join(['%s:%s' % item for item in self.__dict__.items()]))
	return None