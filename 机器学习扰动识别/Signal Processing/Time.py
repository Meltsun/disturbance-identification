#输入原始样本，返回时域特征值。
#当一个特征值需要多行才能算出来时，用单独函数封装。保证time_draw函数够直观
import numpy as np
import math
def time_draw(data:np.narray):
	feature=np.empty(15)
	return feature

#最大值
feature[0]=np.max(data)

#最小值
feature[1]=np.min(data)

#峰峰值(极差)
feature[2]=np.max(data)-np.min(data)

#均值
feature[3]=np.mean(data)

#整流平均值
feature[4]=np.mean(abs(data))

#方差
feature[5]=np.var(data)

#标准差
feature[6]=np.std(data,ddof=1)

#均方根
feature[7]=np.sqrt(np.mean(pow(data,2)))

#能量（对数简化)
feature[8]=10*math.log(np.std(data,ddof=1)*data.size,10)