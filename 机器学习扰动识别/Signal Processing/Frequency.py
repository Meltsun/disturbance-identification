#输入原始样本，返回频域特征值
#当一个特征值需要多行才能算出来时，用单独函数封装。保证frequency_draw函数够直观
import numpy as np
import pywt
x=[1,2,3,4,5,6,7,8]

wp=pywt.WaveletPacket(data=x,wavelet='db1',mode='symmetric',maxlevel=2)

print(wp['aa'].data)
print(wp['aa'].path)
print(len(wp['aa'].data))
print(wp['aa'].data*wp['aa'].data)
LLPE=np.sum(wp['aa'].data*wp['aa'].data)
LHPE=np.sum(wp['ad'].data*wp['ad'].data)
HLPE=np.sum(wp['da'].data*wp['da'].data)
HHPE=np.sum(wp['dd'].data*wp['dd'].data)
E=[LLPE,LHPE,HLPE,HHPE]
P=E/np.sum(E)
WE=-np.sum(np.log2(P)*P)

print(HLPE)
print(HHPE)
print(E)
print(WE)
'''
def frequency_draw(data:np.narray):
	feature=np.empty(15)
	#feature[0]=XXX
	#feature[1]=XXX
	return feature
'''

