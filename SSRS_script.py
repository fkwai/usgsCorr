__author__ = 'KXF227'

import scipy.io as sio
import numpy as np
import SSRS
import matplotlib.pyplot as plt
import random


from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AffinityPropagation
# from sklearn.cluster import Birch

from sklearn import tree
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import svm
from sklearn.linear_model import SGDClassifier

from sklearn import preprocessing

from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn import neural_network

### read data
UCdir = "Y:\Kuai\USGSCorr\\"
UCfile=UCdir+"usgsCorr2.mat"
Datafile=UCdir+"dataset2.mat"
mat = sio.loadmat(UCfile)
UCData=mat['Corr_maxmin']
mat = sio.loadmat(Datafile)
AttrData=mat['dataset']

# ### read departure data
# filename="Y:\Kuai\USGSCorr\dataset_departure.mat"
# mat = sio.loadmat(filename)
# X=mat["X"]
# XX=mat["XX"]

### preprocessing
X = UCData[ind1,]
XX = AttrData[ind2, 0:51]
XX = XX[:, 0:51]
XX[np.isnan(XX)] = 0
scaler = preprocessing.StandardScaler().fit(XX)
XXn = scaler.fit_transform(XX)

### cluster
model = KMeans(init='k-means++', n_clusters=6, n_init=10, max_iter=1000)
model = AffinityPropagation(preference=-150,verbose=True)
model = Birch(branching_factor=10, n_clusters=4, threshold=0.3, compute_labels=True)
model = MeanShift(bandwidth=estimate_bandwidth(X, quantile=0.1, n_samples=100), bin_seeding=True)

label=SSRS.Cluster(X, model)

### classification
model = tree.DecisionTreeClassifier()
model = GaussianNB()
model = svm.SVC()
model = SGDClassifier()

Tp = SSRS.Classification_cross(XXn, T=label, nfold=10, model=model)
SSRS.plotErrorMap(label, Tp)


### regression
regModel=linear_model.LinearRegression()
#regModel=svm.SVC()
regModel=KNeighborsRegressor(n_neighbors=10)
regModel = tree.DecisionTreeRegressor()
regModel = GaussianNB()

rmse_band,Yp,Ytest=SSRS.RegressionLearn(X,XXn,0.2,regModel)

