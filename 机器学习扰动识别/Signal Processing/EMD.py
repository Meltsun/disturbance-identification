import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs

#载入时间序列数据
data = pd.read_csv('C:\\Users\\DELL\\Desktop\\相关资料\\originSet_not_st.csv',header=None).transpose()
decomposer = EMD(data[1]) 
imfs = decomposer.decompose()

#绘制分解
plt.plot([i for i in range(0,len(imfs[0]))],imfs[0]-imfs[1]-imfs[2])
plt.show()
