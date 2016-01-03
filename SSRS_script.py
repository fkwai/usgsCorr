__author__ = 'KXF227'

import scipy.io as sio
import numpy as np
import SSRS

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import Birch

from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import SGDClassifier

from sklearn import preprocessing

# ### read data
# UCdir = "Y:\Kuai\USGSCorr\\"
# UCfile=UCdir+"usgsCorr_maxmin.mat"
# Datafile=UCdir+"dataset2.mat"
# mat = sio.loadmat(UCfile)
# UCData=mat['Corr_maxmin']
# id1=mat['ID']
# id1.reshape(len(id1),)  # change to 1d array in case of further trouble
# mat = sio.loadmat(Datafile)
# AttrData=mat['dataset']
# id2=mat['ID']
# id2.reshape(len(id2),)  # change to 1d array in case of further trouble
#
# id=np.intersect1d(id1,id2)
# ind1=np.zeros((len(id),),int)
# ind2=np.zeros((len(id),),int)
# for i in range(len(id)):
#    ind1[i]=(id[i]==id1).nonzero()[0]
#    ind2[i]=(id[i]==id2).nonzero()[0]

### read departure data
filename="Y:\Kuai\USGSCorr\dataset_departure.mat"
mat = sio.loadmat(filename)
X=mat["X"]
XX=mat["XX"]

### preprocessing
# X=UCData[ind1,]
# XX=AttrData[ind2,0:51]
#XX=XX[:,0:51]
XX[np.isnan(XX)]=0
scaler=preprocessing.StandardScaler().fit(XX)
XXn=scaler.fit_transform(XX)

### cluster
# kmeans
kmeans = KMeans(init='k-means++', n_clusters=6, n_init=10,max_iter=1000)
kmeans.fit(X)
SSRS.clusterplot(X,lable=kmeans.labels_,center=kmeans.cluster_centers_)

# dbscan
db = DBSCAN(eps=0.1, min_samples=50).fit(X)
nclass = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
SSRS.clusterplot(X,lable=db.labels_+1)

# mean shift
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=100)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
print(np.bincount(ms.labels_+1))

# affinity propagation
af = AffinityPropagation(preference=-150,verbose=True)
af.fit(X)
SSRS.clusterplot(X,lable=af.labels_)

# birch
brc = Birch(branching_factor=10, n_clusters=6, threshold=0.5,compute_labels=True)
brc.fit(X)
SSRS.clusterplot(X,lable=brc.labels_)



### training
T=brc.labels_
nfold=10
# Decision Tree
model=tree.DecisionTreeClassifier()
Tp=SSRS.ClusterLearn_cross(XXn,T,nfold,model)
SSRS.plotErrorMap(T,Tp)

# GaussianNB
model=GaussianNB()
Tp=SSRS.ClusterLearn_cross(XXn,T,nfold,model)
SSRS.plotErrorMap(T,Tp)

# SVM
model=svm.SVC()
Tp=SSRS.ClusterLearn_cross(XXn,T,nfold,model)
SSRS.plotErrorMap(T,Tp)

# SGD
model = SGDClassifier()
Tp=SSRS.ClusterLearn_cross(XXn,T,nfold,model)
SSRS.plotErrorMap(T,Tp)