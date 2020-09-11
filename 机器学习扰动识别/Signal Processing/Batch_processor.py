#处理一批数据并保存文件。前期工作的数据处理主函数。

import numpy as np
#打开文件，导入数据
#得到原始数据集originaData（2维列表） 模式集targetSet（1维列表）


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
np.savetxt('地址',allData,delimiter=',')

