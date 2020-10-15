import streamlit as st
import numpy as np
import math
import pandas as pd

#读取整个文件,返回DataFrame
@st.cache
def load_file():
	dsFile = pd.read_csv("originSet.csv",header=None,index_col=None )
	return dsFile

#提取所需要的单个样本，返回ndarray
@st.cache
def load_data(dsType,dsNumber,dsFile):
	ds=dsFile.iloc[dsType*3+dsNumber//50][dsNumber%50*22:dsNumber%50*22+33]
	return ds.values

#提取特征值，返回Series
@st.cache
def get_features(data:np.ndarray):
	featureT={}
	featureT['最大值']=np.max(data)
	featureT['最小值']=np.min(data)
	featureT['峰峰值']=np.max(data)-np.min(data)
	featureT['均值']=np.mean(data)
	featureT["整流均值"]=np.mean(abs(data))
	featureT["均方根"]=np.sqrt(np.mean(pow(data,2)))
	featureT["能量对数"]=10*math.log(np.var(data,ddof=1)*data.size,10)
	featureT["方差"]=np.var(data)
	featureT["标准差"]=np.std(data,ddof=1)
	featureT["波形特征"]=np.mean(pow(data,4))/pow(featureT["均方根"],4)
	featureT["偏度"]=np.mean(pow(((data-featureT['均值'])/np.sqrt(featureT["方差"])),3))
	featureT["裕度"]=featureT['最大值']/pow(np.mean(np.sqrt(abs(data))),2)
	featureT["波形"]=featureT["均方根"]/featureT["整流均值"]
	featureT['峰值因子1']=featureT['最大值']/featureT["均方根"]
	featureT['峰值因子2']=featureT['最大值']/featureT["整流均值"]
	featureT['(暂缺)']=None
	#差分 特征值
	dataDiff=np.diff(data)
	featureD={}
	featureD["最大值"]=np.max(dataDiff)
	featureD['最小值']=np.min(dataDiff)
	featureD['极差']=np.max(dataDiff)-np.min(dataDiff)
	featureD['均值']=np.mean(dataDiff)
	featureD['整流均值']=np.mean(abs(dataDiff))
	featureD['均方根']=np.sqrt(np.mean(pow(dataDiff,2)))
	featureD['能量对数']=10*math.log(np.var(dataDiff,ddof=1)*dataDiff.size,10)
	featureD['方差']=np.var(dataDiff)
	featureD['标准差']=np.std(dataDiff,ddof=1)
	featureD['峭度']=np.mean(pow(dataDiff,4))/pow(featureD['均方根'],4)
	featureD['偏度']=np.mean(pow(((dataDiff-featureD['均值'])/np.sqrt(featureD['方差'])),3))*10
	featureD['裕度']=featureD["最大值"]/pow(np.mean(np.sqrt(abs(dataDiff))),2)
	featureD['波形特征']=featureD['均方根']/featureD['整流均值']
	featureD['峰值因子']=featureD["最大值"]/featureD['均方根']
	featureD['脉冲因子']=featureD["最大值"]/featureD['整流均值']
	featureD['(暂缺)']=None
	return pd.Series(featureT),pd.Series(featureD),pd.Series(dataDiff)

#定义通用的变量
dsTypeN = ['无','车辆经过','浇水','敲击','攀爬']
dsCruveN = ['时域','时域差分','小波分解','经验模态分解']

dsFile=load_file()

st.title('分布式光纤传感系统的扰动信号分析')
agree = st.sidebar.radio("分析模式",['单个样本','一类样本'])

if(agree=='单个样本'):
	#获取侧栏的控制字
	dsType = dsTypeN.index(st.sidebar.selectbox('扰动类型' , dsTypeN))#整数，0-4
	dsNumber = st.sidebar.slider('样本编号',1,150,50,1)-1#整数，0-149
	dsCruve = st.sidebar.selectbox('图像类型' , dsCruveN)#字符串
	
	#单个样本分析中的通用变量
	ds=load_data(dsType,dsNumber,dsFile)
	featureT,featureD,dsDiff=get_features(ds)

	#标题
	st.title(f'【{dsType}扰动信号No.{dsNumber}】- {dsCruve} ')
	
	if(dsCruve=='时域'):
		ds
		st.dataframe(pd.DataFrame(featureT.iloc[:8],columns=['*']).T)
		st.dataframe(pd.DataFrame(featureT.iloc[8:],columns=['*']).T)

	elif(dsCruve=='时域差分'):
		dsDiff
		st.dataframe(pd.DataFrame(featureD.iloc[:8],columns=['*']).T)
		st.dataframe(pd.DataFrame(featureD.iloc[8:],columns=['*']).T)

	else:
		'（还没做好 -_-||）'

elif(agree=='一类样本'):
	dsCruve = st.sidebar.selectbox('横轴特征值' , [1,2,3,4,5])#字符串
#st.text('感觉不对劲请点击右上角‘Clear cache’重新加载')
#st.text('by BJTU_WXY')