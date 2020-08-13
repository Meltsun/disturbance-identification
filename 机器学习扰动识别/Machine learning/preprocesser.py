from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle

class dataset:
	nFeatures=30 #这是一个人工确定的值
	nSample=250 
	target=None
	data=None
	nLable={}

	#构造方法
	def __init__(self,N=0):
		if(N!=0):
			self.nSample=N
		self.target=np.empty(N)
		self.data=np.empty([N,self.nFeatures])
		return None

	#从字符串列表生成数据集
	def build_from_file(self,file):
		self.nSample=len(file)
		self.target=np.empty(self.nSample)
		self.data=np.empty([self.nSample,self.nFeatures])
		for i in range(0,self.nSample):
			thisData=[eval(t) for t in file[i].split("\t")]
			thisTarget=thisData[-1]
			self.target[i]=thisTarget
			self.data[i]=thisData[:-1]
			if thisTarget in self.nLable.keys(): 
				self.nLable[thisTarget]+=1
			else:
				self.nLable[thisTarget]=1
		return None

	#将自身数据标准化，目前为z-score标准化
	def zscore_standardize(self):
		self.data=preprocessing.StandardScaler().fit_transform(self.data)
		return None

	def minmax_standardize(self):
		self.data=preprocessing.MinMaxScaler().fit_transform(self.data)
		return None

	#切分数据并随机打乱,输入为前一返回数据集的占比
	def split(self,sampleProportion): 
		dataset1=dataset()
		dataset2=dataset()
		dataset1.data,dataset2.data,dataset1.target,dataset2.target=train_test_split(self.data,self.target, test_size=sampleProportion)
		return dataset2,dataset1

	#根据条件生成数据集,输入为标签为所有标签为0的动作和所有为1的，返回生成的数据集
	def conditional_extract(self,condition0,condition1): 
		if(type(condition0)==type(0)):
			condition0=[condition0]
		if(type(condition1)==type(1)):
			condition1=[condition1]
		N=0
		thisLable={}
		for i in condition0+condition1:
			N+=self.nLable[i]
			thisLable[i]=self.nLable[i]
		thisDataset=dataset(N)
		thisDataset.nLable=thisLable
		N=0
		for i in range(0,self.nSample):
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
	
	def print(self): 
		print('\n'.join(['%s:%s' % item for item in self.__dict__.items()]))

	def shuffle(self):
		self.data,self.target=shuffle(self.data,self.target)
#数据导入和预处理，返回一个数据集

def preprocess(): 
	file=open("D:\大创\处理后的数据.txt")
	file1=file.readlines()
	file.close()
	totalData=dataset()
	totalData.build_from_file(file1)
	return totalData


