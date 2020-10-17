#输入原始样本，返回时域特征值。
#当一个特征值需要多行才能算出来时，用单独函数封装。保证time_draw函数够直观
import numpy as np
import math

def time_draw(data:np.ndarray):
	data2=np.diff(data)
	feature=np.empty(30)
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
	feature[6]=10*math.log(np.var(data,ddof=1)*data.size,10)

	#离散程度特征
	#方差
	feature[7]=np.var(data)

	#标准差
	feature[8]=np.std(data,ddof=1)

	#波形特征
	#峭度（波形的平缓程度）
	feature[9]=np.mean(pow(data,4))/pow(feature[5],4)

	#偏度（数据分布非对称程度）
	feature[10]=np.mean(pow(((data-feature[3])/np.sqrt(feature[7])),3))

	#裕度因子（对冲击敏感）
	feature[11]=feature[0]/pow(np.mean(np.sqrt(abs(data))),2)

	#波形因子（对峰值敏感）
	feature[12]=feature[5]/feature[4]

	#峰值因子
	feature[13]=feature[0]/feature[5]

	#脉冲因子
	feature[14]=feature[0]/feature[4]

	#幅值特征
	#最大值
	feature[15]=np.max(data2)

	#最小值
	feature[16]=np.min(data2)

	#峰峰值(极差)
	feature[17]=np.max(data2)-np.min(data2)

	#均值
	feature[18]=np.mean(data2)

	#整流平均值
	feature[19]=np.mean(abs(data2))

	#均方根
	feature[20]=np.sqrt(np.mean(pow(data2,2)))

	#能量（对数简化)
	feature[21]=10*math.log(np.var(data2,ddof=1)*data2.size,10)

	#离散程度特征
	#方差
	feature[22]=np.var(data2)

	#标准差
	feature[23]=np.std(data2,ddof=1)

	#波形特征
	#峭度（波形的平缓程度）
	feature[24]=np.mean(pow(data2,4))/pow(feature[20],4)

	#偏度（数据分布非对称程度）
	feature[25]=np.mean(pow(((data2-feature[18])/np.sqrt(feature[22])),3))*10

	#裕度因子（对冲击敏感）
	feature[26]=feature[15]/pow(np.mean(np.sqrt(abs(data2))),2)

	#波形因子（对峰值敏感）
	feature[27]=feature[20]/feature[19]

	#峰值因子
	feature[28]=feature[15]/feature[20]

	#脉冲因子
	feature[29]=feature[15]/feature[19]


	return feature

