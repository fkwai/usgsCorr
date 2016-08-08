import scipy.io as sio
import scipy
import matplotlib.pyplot as plt
import numpy as np
import itertools
from time import clock

from sklearn import preprocessing
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA

from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor

from sklearn import cross_validation

import SSRS

UCdir = "E:\work\SSRS\data\\"
UCfile=UCdir+"usgsCorr_mB_4949.mat"
Datafile=UCdir+"dataset_mB_4949.mat"

mat = sio.loadmat(UCfile)
UCData=mat['usgsCorr']
mat = sio.loadmat(Datafile)
AttrData=mat['dataset']
Field = [str(''.join(letter)) for letter_array in mat['field'] for letter in letter_array]

Field=[Field[i] for i in range(0,52)]
Y=UCData
attrind=np.array(range(0,52))
X=AttrData[:,attrind]

#remove nan
indnanX=np.unique(np.where(np.isnan(X))[0])
indnanY=np.unique(np.where(np.isnan(Y))[0])
indnan=np.unique(np.concatenate((indnanX,indnanY), axis=0))
indvalid=np.array(range(0,X.shape[0]))
X=np.delete(X,indnan,0)
Y=np.delete(Y,indnan,0)
indvalid=np.delete(indvalid,indnan,0)

scaler=preprocessing.StandardScaler().fit(X)
Xn=scaler.fit_transform(X)
[nind,nband]=Y.shape
[nind,nattr]=X.shape
