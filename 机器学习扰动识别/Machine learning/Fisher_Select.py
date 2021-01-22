#Fisher得分法筛选特征值
#输入训练集，返回得分由高到低排序的40个特征值对应的序号

import numpy as np
import matplotlib.pyplot as plt

def Fisher_Select(traindata:np,targetdata:np):
	
	
	Fisher=np.empty(40)
	data_mean=np.empty([5,40])
	data_var = np.empty([5,40])
	
	#将训练集5类数据分开
	npose = 5
	nsmile = 2

	data = np.empty((npose,nsmile),dtype=object)
	
	for i in range(5):
		for k in range(2):
			target = targetdata
			target == i
			data[i][k] = traindata[target == i]
 
	#计算训练集每类特征值的均值与方差
	for i in range(0,4):
		data_mean[i][:]=np.mean(data[i][0],axis=0)
		data_var[i][:]=np.var(data[i][0],axis=0)
		
	#利用Fisher得分法计算40个特征值得分
	for i in range(0,39):
		for j in range(0,4):
			for k in range(j+1,4):
				Fisher[i] = Fisher[i]+((data_mean[j][i]-data_mean[k][i])**2)/(data_var[j][i]+data_var[k][i])

	y = np.sort(Fisher,axis = 0)
	idx = np.argsort(-Fisher) # 逆序输出索引，从大到小
	print("Fisher得分结果\n")
	print(idx) #打印结果

	#画图测试
	x=np.empty(40)
	for i in range(0,39):
		x[i]=i
	plt.bar(x=x, height=Fisher)
	plt.show()
	
	return idx