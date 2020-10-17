import streamlit as st
import numpy as np
import pandas as pd
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


#读取整个文件,返回ndarray
@st.cache
def load_dataset():
	return pd.read_csv("originSet.csv",header=None,index_col=None ).values

#提取特征值DataFrame
@st.cache
def load_features():
	return pd.read_csv("feature.csv",index_col=None,header=0,encoding='gbk')

#定义通用的变量
dataSet=load_dataset()#所有数据
feature=load_features()#所有特征值
dsTypeN = ['无','车辆经过','浇水','敲击','攀爬']
dsCruveN = ['时域','时域差分','小波包分解','经验模态分解']
featureN=feature.columns#特征值名


st.title('分布式光纤传感系统的扰动信号分析')
agree = st.sidebar.radio("分析模式",['单个样本','一类样本'])
st.sidebar.write('---')

if(agree=='单个样本'):
	#获取侧栏的控制字
	dsType = st.sidebar.selectbox('扰动类型' , dsTypeN)
	
	dsNumber = st.sidebar.slider('样本编号',1,150,50,1)-1#整数，0-149
	st.sidebar.write('---')
	dsCruve = st.sidebar.selectbox('图像类型' , dsCruveN)#字符串
	
	#小标题
	st.title(f'【{dsType}扰动信号No.{dsNumber}】- {dsCruve} ')

	dsType = dsTypeN.index(dsType)#整数，0-4
	#单个样本分析中的通用变量
	data=dataSet[dsType*3+dsNumber//50][dsNumber%50*22:dsNumber%50*22+33]
	
	#dFeature
	if(dsCruve=='时域'):
		#ser=pd.Series(data.tolist())
		#ser.plot()
		st.line_chart(data)
		#st.pyplot()
		#图
		dFeature=feature.iloc[dsType*150+dsNumber][:-1][0:15]
		st.dataframe(pd.DataFrame(dFeature[0:8]).T)
		st.dataframe(pd.DataFrame(dFeature[8:]).T)

	elif(dsCruve=='时域差分'):
		dataDiff=np.diff(data)
		#图
		figure1=plt.figure()
		plt.plot([i for i in range(0,len(dataDiff))],dataDiff)
		st.pyplot(figure1)

		dFeature=feature.iloc[dsType*150+dsNumber][:-1][15:30]
		dFeature.index=[i.replace('差分-','') for i in dFeature.index]
		st.dataframe(pd.DataFrame(dFeature[0:8]).T)
		st.dataframe(pd.DataFrame(dFeature[8:]).T)

	elif(dsCruve=='小波包分解'):
		dFeature=feature.iloc[dsType*150+dsNumber][:-1][30:40]
		st.dataframe(pd.DataFrame(dFeature[0:5]).T)
		st.dataframe(pd.DataFrame(dFeature[5:]).T)
		dsCruveWP = st.sidebar.selectbox('选择小波图像' , ['最低频','低频','高频','最高频'])
		#根据dsCruveWP的值（字符串）画不同的图

	else:
		'（还在做 -_-||）'


elif(agree=='一类样本'):
	#获取各种控制字
	featureX = st.sidebar.selectbox('X' , featureN)#字符串
	featureY = st.sidebar.selectbox('Y' , [i for i in featureN if i !=featureX])#字符串
	st.sidebar.write('---')
	dsShow = [st.sidebar.checkbox(dsTypeN[i]) for i in range(0,5)]
	dsShow = [i for i in range(0,5) if dsShow[i]]#获得被所有勾选的类型编号组成的列表
	#根据选择的类别和特征值画散点图:
	'（还在做 -_-||）'

	



st.write('---')
st.text('感觉不对劲请点击右上角‘Clear cache’重新加载')
st.text('by BJTU_WXY')