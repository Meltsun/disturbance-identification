#Fisher得分法筛选特征值
#输入训练集，返回得分由高到低排序的40个特征值对应的序号

import numpy as np
import matplotlib.pyplot as plt

def Fisher_Select(traindata:np):
	
	row_num = traindata.shape[0]
	Fisher=np.empty(40)
	data_mean=np.empty([5,41])
	data_var = np.empty([5,41])

	target = np.empty(row_num)
	target = traindata[:,-1]
	target == 0
	data0 = traindata[target == 0]
	
	target = traindata[:,-1]
	target == 1
	data1 = traindata[target == 1]
	
	target = traindata[:,-1]
	target == 2
	data2 = traindata[target == 2]
	
	target = traindata[:,-1]
	target == 3
	data3 = traindata[target == 3]
	
	target = traindata[:,-1]
	target == 4
	data4 = traindata[target == 4]
	
	data_mean[0][:]=np.mean(data0,axis=0)
	data_var[0][:]=np.var(data0,axis=0)

	data_mean[1][:]=np.mean(data1,axis=0)
	data_var[1][:]=np.var(data1,axis=0)

	data_mean[2][:]=np.mean(data2,axis=0)
	data_var[2][:]=np.var(data2,axis=0)

	data_mean[2][:]=np.mean(data2,axis=0)
	data_var[2][:]=np.var(data2,axis=0)

	data_mean[3][:]=np.mean(data3,axis=0)
	data_var[3][:]=np.var(data3,axis=0)

	data_mean[4][:]=np.mean(data4,axis=0)
	data_var[4][:]=np.var(data4,axis=0)

	for i in range(0,39):
		for j in range(0,4):
			for k in range(j+1,4):
				Fisher[i] = Fisher[i]+abs(((data_mean[j][i]-data_mean[k][i])**2)/(data_var[j][i]-data_var[k][i]))

	x=np.empty(40)
	for i in range(0,39):
		x[i]=i

	y = np.sort(Fisher,axis = 0)
	idx = np.argsort(-Fisher) # 逆序输出索引，从大到小

	#画图
	plt.bar(x=x, height=Fisher)
	plt.show()

	
	return idx