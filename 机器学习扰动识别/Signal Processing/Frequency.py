#输入原始样本，返回频域特征值
#当一个特征值需要多行才能算出来时，用单独函数封装。保证frequency_draw函数够直观
import numpy as np
import pywt

def frequency_draw(data2:np.ndarray):
	wp=pywt.WaveletPacket(data=data2,wavelet='db3',mode='symmetric',maxlevel=2) #2层小波包分解
	new_wp = pywt.WaveletPacket(data=None, wavelet='db3', mode='symmetric',maxlevel=2) #新建小波包树用来重构4个节点系数

	new_wp['aa'] = wp['aa']
	LL=new_wp.reconstruct(update=False)


	del(new_wp['aa'])
	new_wp['ad'] = wp['ad']
	LH=new_wp.reconstruct(update=False)
	

	del(new_wp['a'])
	new_wp['da'] = wp['da']
	HL=new_wp.reconstruct(update=False)


	del(new_wp['da'])
	new_wp['dd'] = wp['dd']
	HH=new_wp.reconstruct(update=False)

	
	#4个频率分量的能量
	LLPE=np.sum(LL*LL)    
	LHPE=np.sum(LH*LH)
	HLPE=np.sum(HL*HL)
	HHPE=np.sum(HH*HH)

	E=[LLPE,LHPE,HLPE,HHPE]
	P=E/np.sum(E)
	WE=-np.sum(np.log2(P)*P)	#信息熵
	WIQ=np.sum(P*E)	#信息量
	S=np.array([LL,LH,HL,HH])	#小波系数矩阵4*33
	U,sigma,VT=np.linalg.svd(S,full_matrices=1,compute_uv=1)	#奇异值分解

	feature=np.empty(10)

	feature[0]=LLPE	#低频中低频部分的能量

	feature[1]=LHPE	#低频中高频部分的能量

	feature[2]=HLPE	#高频中低频部分的能量

	feature[3]=HHPE	#高频中高频部分的能量

	feature[4]=WE	#小波信息熵

	feature[5]=WIQ	#小波信息量

	feature[6]=sigma[0]	#奇异值1

	feature[7]=sigma[1]	#奇异值2

	feature[8]=sigma[2]	#奇异值3

	feature[9]=sigma[3]	#奇异值4

	return feature





