import os
import scipy.io as sio
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import itertools
from textwrap import wrap

from sklearn import metrics
from math import sqrt

import sklearn
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import manifold
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
#from sknn.mlp import Regressor, Layer
from sklearn.decomposition import PCA


import SSRS

### read data
UCdir = "Y:\Kuai\USGSCorr\\"
#UCdir = r"/Volumes/wrgroup/Kuai/USGSCorr/"
UCfile=UCdir+"usgsCorr2.mat"
Datafile=UCdir+"dataset3.mat"
mat = sio.loadmat(UCfile)
UCData=mat['Corr_maxmin']
mat = sio.loadmat(Datafile)
AttrData=mat['dataset']
Field = [str(''.join(letter)) for letter_array in mat['field'] for letter in letter_array]

## preprocessing
Y=UCData
attrind=np.array(range(1,51)+range(62,78,3))
Field=[Field[i] for i in range(1,51)+range(62,78,3)]
X=AttrData[:,attrind]
X[np.isnan(X)]=0
scaler=preprocessing.StandardScaler().fit(X)
Xn=scaler.fit_transform(X)

##best band
Y1=np.argmax(Y[:,0:15],axis=1)
Y2=np.argmax(Y[:,15:30],axis=1)

## Regression
regModel=linear_model.LinearRegression()
#regModel=svm.SVC()
regModel=KNeighborsRegressor(n_neighbors=20)
regModel=tree.DecisionTreeRegressor()
regModel=GaussianNB()
regModel=sklearn.linear_model.SGDRegressor()
regModel=RandomForestRegressor()

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(\
        Xn,np.column_stack((Y1,Y2)),test_size=0.2,random_state=0)

Yp,rmse,rmse_train,rmse_band,rmse_band_train=SSRS.Regression\
    (X_train,X_test,Y_train,Y_test,multiband=1,regModel=regModel,doplot=0)
print(rmse)
print(rmse_train)

## Classification
model = tree.DecisionTreeClassifier()
model = GaussianNB()
model = svm.SVC()
model = SGDClassifier()
model=sklearn.ensemble.RandomForestClassifier()

Yin=Y1
Tp = SSRS.Classification_cross(Xn, T=Yin, nfold=10, model=model)
SSRS.plotErrorMap(Yin, Tp)
np.sqrt(((Yin - Tp) ** 2).mean())
np.count_nonzero(np.abs(Yin-Tp)<2)/4627.