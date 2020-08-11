"""
提供数据集类
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class dataset:
    target=np.empty(0)
    data=np.empty(0)

    def __init__(self):
        return None

    def build_from_file(self,file):#从字符串列表生成数据集
        return None

    def standardized(self):#标准化，不同的算法可能对应多个标准化函数
        return None

    def split(self,sampleProportion): #切分数据,输入为前一返回数据集的占比
        dataset1=dataset()
        dataset2=dataset()
        dataset1.data,dataset2.data,dataset1.target,dataset2.target=train_test_split(self.data,self.target, test_size=sampleProportion)

        return dataset2,dataset1                   
       
    
  



