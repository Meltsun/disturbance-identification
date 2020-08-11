"""
main
广义神经网络-预测器
参数已确定情况下，输出预测准确率

"""


import numpy as np               #基本计算
from preprocesser import dataset #数据集类

#导入数据文件
file=open("")
file1=file.readlines()
file.close()

#生成数据集
totalData=dataset().build_from_file(file1)

#训练
trainData,testData=totalData.split(0.2)
Classifier=None

#验证，输出正确率
forecastResults=Classifier.predict(testData.data)
nRight=0
for i in range(0,len(testData.target)):
    if(testData.target[i]==forecastResults[i]):
        nRight+=1
print(f"正确率：{100*nRight/len(testData.target):2f}%")
        











新奇新奇辛奇


   




