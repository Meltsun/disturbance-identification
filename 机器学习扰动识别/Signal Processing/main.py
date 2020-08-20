import scipy.io as scio


datafile = 'datajiao.mat'

datajiao = scio.loadmat(datafile)
print(type(datajiao))#字典