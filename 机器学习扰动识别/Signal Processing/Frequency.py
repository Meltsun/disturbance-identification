#输入原始样本，返回频域特征值
#当一个特征值需要多行才能算出来时，用单独函数封装。保证frequency_draw函数够直观
import numpy as np
import pywt

def frequency_draw(data2:np.ndarray):
	wp=pywt.WaveletPacket(data=data2,wavelet='db3',mode='symmetric',maxlevel=2)
	LLPE=np.sum(wp['aa'].data*wp['aa'].data)
	LHPE=np.sum(wp['ad'].data*wp['ad'].data)
	HLPE=np.sum(wp['da'].data*wp['da'].data)
	HHPE=np.sum(wp['dd'].data*wp['dd'].data)
	E=[LLPE,LHPE,HLPE,HHPE]
	P=E/np.sum(E)
	WE=-np.sum(np.log2(P)*P)
	WIQ=np.sum(P*E)
	S=np.array([wp['aa'].data,wp['ad'].data,wp['da'].data,wp['dd'].data])
	U,sigma,VT=np.linalg.svd(S,full_matrices=1,compute_uv=1)
	feature=np.empty(10)
	feature[0]=LLPE #低频的低频能量
	feature[1]=LHPE#低频的高频
	feature[2]=HLPE#高频的低频
	feature[3]=HHPE#高频的高频
	feature[4]=WE  #信息熵
	feature[5]=WIQ   #信息量
	#四个奇异值
	feature[6]=sigma[0]   
	feature[7]=sigma[1]
	feature[8]=sigma[2]
	feature[9]=sigma[3]

	return feature

x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
s=frequency_draw(x)
print(s)



