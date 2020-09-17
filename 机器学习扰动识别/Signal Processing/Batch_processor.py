#处理一批数据并保存文件。前期工作的数据处理主函数。
from Frequency import frequency_draw
from Time import time_draw
from sklearn import preprocessing
import numpy as np

##划分,得到样本集data（2维numpy数组）
T=33#周期，必须为33的倍数。
nTime=1111#每个点的时域数据数
nSample=(nTime-T//3)//(T*2//3)#每个点取样本数量
data=np.empty([nSample*15,T])
targetSet=np.empty(nSample*15)

originSet=open("originSet.csv").readlines()

for i in range(0,15):
	thisData=originSet[i].split(',')
	originSet[i]=[eval(n) for n in thisData]

for i in range(0,15):
	for j in range(0,nSample):
		data[i*nSample+j]=originSet[i][j*(T*2//3):j*(T*2//3)+33]
		targetSet[i*nSample+j]=i//3
print("样本划分成功")
#调用特征值提取函数，得到样本特征集

#from Time import time_draw
nTimeFeature=15
nFrequencyFeature=10
featureSet=np.empty([nSample*15,nTimeFeature+nFrequencyFeature])

for i in range(0,nSample*15):
	featureSet[i][:nTimeFeature]=time_draw(data[i])
	featureSet[i][nTimeFeature:]=frequency_draw(data[i])
print("特征值提取成功")
#保存为文件
allData=np.empty([nSample*15,nTimeFeature+nFrequencyFeature+1])
#allData[:,:-1]=preprocessing.MinMaxScaler().fit_transform(featureSet)
allData[:,:-1]=featureSet
allData[:,-1]=targetSet
np.savetxt('feature.csv',allData,delimiter=',',fmt='%f')


