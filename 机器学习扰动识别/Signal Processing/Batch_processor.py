#处理一批数据并保存文件。前期工作的数据处理主函数。

points=[35,259,859]
norms=[131626.41826016537, 348089.33621126635, 171083.7945803167, 532644.7420636009, 240586.45893732258] 
nTime=1001

import numpy as np
#打开文件，导入数据
#得到原始数据集originaSet（2维列表） 模式集targetSet（1维列表）
file=[open("data_Nothing.csv").readlines(),open("data_Pass.csv").readlines(),open("data_Water.csv").readlines(),open("data_Knock.csv").readlines(),open("data_Climb.csv").readlines()]
originSet=np.empty([15,nTime])

for j in range(0,len(file)):
	thisData=np.empty([nTime,1600])
	for i in range(0,nTime):
		thisData[i]=np.array([eval(t) for t in file[j][i].split(",")])
	for i in range(0,len(points)):
		x=thisData[:,points[i]]
		originSet[len(points)*j+i]=-x/np.linalg.norm(x)
	

#归一化，划分

#得到原始样本集data（2维numpy数组） 


#调用特征值提取函数，得到样本特征集
from Frequency import frequency_draw,time_draw
featureSet=np.empty([nSample,nTimeFeature+nFrequencyFeature])
for i in range(0,nSample):
	featureSet[i][:nTimeFeature]=time_draw(data[i])
	featureSet[i][nTimeFeature:]=frequency_draw(data[i])

#保存为文件
allData=np.empty([nSample,nTimeFeature+nFrequencyFeature+1])
for i in range(0,nSample):
	allData[:,:-1]=featureSet
	allData[:,-1]=targetSet
np.savetxt('feature.csv',allData,delimiter=',')

