import streamlit as st
import numpy as np
from Time import time_draw
from Frequency import frequency_draw
import pandas as pd

@st.cache
def load_file():
    ds = pd.read_csv("originSet.csv",header=None,index_col=None )
    return ds

@st.cache
def load_data(dsType,dsNumber):
    ds=load_file()
    ds=ds.iloc[dsType*3+dsNumber//50][dsNumber%50*22:dsNumber%50*22+33]
    return ds

@st.cache
def get_features(ds):
    ds=ds.values
    features=np.empty(40)
    features[:30],dsDiff=time_draw(ds)
    features[30:]=frequency_draw(ds)
    return pd.DataFrame(features),dsDiff

st.title('分布式光纤传感系统的扰动信号分析')
agree = st.sidebar.radio("分析模式",['单个样本','一类样本'])

dsTypeN = ['无','车辆经过','浇水','敲击','攀爬']
dsCruveN = ['时域','时域差分','小波分解','经验模态分解']
featureTN = ['最大值','最小值','峰峰值','均值','整流均值','均方根',
             '能量（对数简化）','方差','标准差','翘度','偏度',
             '裕度因子','波形因子','峰值因子','脉冲因子']
featureFN = [] 

if(agree=='单个样本'):
    #获取侧栏的控制字
    dsType = st.sidebar.selectbox('扰动类型' , dsTypeN)
    dsNumber = st.sidebar.slider('样本编号',1,150,50,1)
    dsCruve = st.sidebar.selectbox('图像类型' , dsCruveN)    
    st.title(f'【{dsType}扰动信号No.{dsNumber}】- {dsCruve} ')

    dsType = dsTypeN.index(dsType)
    dsNumber -= 1
    
    ds=load_data(dsType,dsNumber)

    features , dsDiff = get_features(ds)

    if(dsCruve=='时域'):
        ds
        features=features[:15]
        features.index=featureTN
        st.dataframe(features)

    elif(dsCruve=='时域差分'):
        dsDiff
        features=features[15:30]
        features.index=featureTN
        st.dataframe(features)

    else:
        '（还没做好 -_-||）'
    