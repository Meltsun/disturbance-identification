import scipy.io as scio
datafile = 'datajiao.mat'
datajiao = scio.loadmat(datafile)
datajiao=datajiao['datajiao']
print(datajiao)

import numpy as np
jiaoguiyihua=[]
for i in datajiao:
    i_norm=np.linalg.norm(i, ord=2)
    for x in i:
        fan=x/i_norm
        jiaoguiyihua.append(fan)
print(jiaoguiyihua)
