from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

	def build_from_file(self,file):#从字符串列表生成数据集
		self.nSample=len(file)
		self.target=np.empty(self.nSample)
		self.data=np.empty([self.nSample,self.nFeatures])
		for i in range(0,self.nSample):
			thisData=[eval(t) for t in file[i].split("\t")]
			thisTarget=thisData[-1]
			self.target[i]=thisTarget
			self.data[i]=thisData[:-1]
			if thisTarget in nLable: 
				nLable[thisTarget]+=1
			else:
				nLable[thisTarget]=1
		return None

	def standardize(self):#标准化，目前为z-score标准化
		self.data=StandardScaler().fit_transform(self.data)
		return None

	def split(self,sampleProportion): #切分数据并随机打乱,输入为前一返回数据集的占比
		dataset1=dataset()
		dataset2=dataset()
		dataset1.data,dataset2.data,dataset1.target,dataset2.target=train_test_split(self.data,self.target, test_size=sampleProportion)
		return dataset2,dataset1

	def conditional_extract(self,condition0,condition1): #根据条件生成数据集
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
			elif(thisLable in condition1):
				thisDataset.data[N]=self.data[i]
				thisDataset.target[N]=1
		return 
		
def preprocess(): 
	file=open("D:\大创\处理后的数据.txt")
	file1=file.readlines()
	file.close()
	totalData=dataset().build_from_file(file1)

preprocess()





