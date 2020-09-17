#处理一批数据并保存文件。前期工作的数据处理主函数。

points=[35,259,859]
norms=[131626.41826016537, 348089.33621126635, 171083.7945803167, 532644.7420636009, 240586.45893732258] 
nTime=1001#每个点的时域数据数

import numpy as np
#打开文件，导入数据
file=[open("data_Nothing.csv").readlines(),open("data_Pass.csv").readlines(),open("data_Water.csv").readlines(),open("data_Knock.csv").readlines(),open("data_Climb.csv").readlines()]
originSet=np.empty([15,nTime])

##得到原始数据集originaSet（2维列表） 模式集targetSet（1维列表
for j in range(0,len(file)):
	thisData=np.empty([nTime,1600])
	for i in range(0,nTime):
		thisData[i]=np.array([eval(t) for t in file[j][i].split(",")])
	for i in range(0,len(points)):
		x=thisData[:,points[i]]
		originSet[len(points)*j+i]=-x/np.linalg.norm(x)
	
#划分,得到样本集data（2维numpy数组）
T=33#周期，必须为33的倍数。
nSample=(nTime-T//3)//(T*2//3)#每个点取样本数量
data=np.empty([nSample*15,T])
targrt=np.empty(nSample*15)

for i in range(0,15):
	for j in range(0,nSample):
		data[i*nSample+j]=originSet[i][j*(T*2//3):j*(T*2//3)+33]
		targrt[i*nSample+j]=i//3

#调用特征值提取函数，得到样本特征集
from Frequency import frequency_draw
from Time import time_draw
nTimeFeature=0
nFrequencyFeature=0
featureSet=np.empty([nSample*15,nTimeFeature+nFrequencyFeature])
for i in range(0,nSample):
	featureSet[i][:nTimeFeature]=time_draw(data[i])
	featureSet[i][nTimeFeature:]=frequency_draw(data[i])

#保存为文件
allData=np.empty([nSample,nTimeFeature+nFrequencyFeature+1])
for i in range(0,nSample):
	allData[:,:-1]=featureSet
	allData[:,-1]=targetSet
np.savetxt('feature.csv',allData,delimiter=',')

