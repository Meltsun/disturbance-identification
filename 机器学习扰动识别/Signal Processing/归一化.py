import scipy.io as scio
datafile = 'datajiao.mat'
datajiao = scio.loadmat(datafile)
print(datajiao)

import numpy as np  
arr = np.asarray(datajiao)
for x in arr:
  x = float(x - arr.mean())/arr.std()
  print(x)
