#输入原始样本，返回时域特征值。
#当一个特征值需要多行才能算出来时，用单独函数封装。保证time_draw函数够直观
import numpy as np
import math
def time_draw(data:np.ndarray):
	feature=np.empty(15)
	#幅值特征
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

	#均方根
	feature[5]=np.sqrt(np.mean(pow(data,2)))

	#能量（对数简化)
	feature[6]=10*math.log(np.std(data,ddof=1)*data.size,10)

	#离散程度特征
	#方差
	feature[7]=np.var(data)

	#标准差
	feature[8]=np.std(data,ddof=1)

	#波形特征
	#峭度（波形的平缓程度）
	feature[9]=np.sqrt(np.mean(pow(data,4))/feature[5])
	
	#偏度（数据分布非对称程度）
	feature[10]=np.mean(pow(((data-feature[3])/np.sqrt(feature[7])),3))

	#裕度因子（对冲击敏感）
	feature[11]=feature[2]/pow(np.mean(np.sqrt(abs(data))),2)

	#波形因子（对峰值敏感）
	feature[12]=feature[5]/feature[4]

	#峰值因子
	feature[13]=feature[2]/feature[5]

	#脉冲因子
	feature[14]=feature[2]/feature[4]

	return feature
