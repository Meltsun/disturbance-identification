points=[35,259,859]
nTime=1111#每个点的时域数据数

import numpy as np
#打开文件，导入数据
file=[open("data_Nothing.csv").readlines(),open("data_Pass.csv").readlines(),open("data_Water.csv").readlines(),open("data_Knock.csv").readlines(),open("data_Climb.csv").readlines()]
originSet=np.empty([len(file)*len(points),nTime])
print("csv文件读取成功")

##得到原始数据集originaSet（2维列表） 模式集targetSet（1维列表
for j in range(0,len(file)):
	thisData=np.empty([nTime,1600])
	for i in range(0,nTime):
		thisData[i]=np.array([eval(t) for t in file[j][i].split(",")])
	for i in range(0,len(points)):
		x=thisData[:,points[i]]
		originSet[len(points)*j+i]=x/np.linalg.norm(x)*40
np.savetxt('originSet.csv',originSet,delimiter=',',fmt='%f')

print("样本原始数据导入成功")
