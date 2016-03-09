import scipy.io as sio
import scipy
import matplotlib.pyplot as plt
import numpy as np

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

## read data
UCdir = "Y:\Kuai\USGSCorr\\"
UCfile=UCdir+"usgsCorr2.mat"
Datafile=UCdir+"dataset2.mat"
mat = sio.loadmat(UCfile)
UCData=mat['Corr_maxmin']
mat = sio.loadmat(Datafile)
AttrData=mat['dataset']
field=mat['field']
figdir='Y:\\Kuai\\USGSCorr\\figures_dist\\'

## preprocessing
Y=UCData
X=AttrData[:,0:51]
X[np.isnan(X)]=0
scaler=preprocessing.StandardScaler().fit(X)
Xn=scaler.fit_transform(X)

[nind,nband]=Y.shape
[nind,nattr]=X.shape

## cluster
nc=6
model = KMeans(init='k-means++', n_clusters=nc, n_init=10, max_iter=1000)
label,center=SSRS.Cluster(Y, model,doplot=0)


# ## manifold take long time
# Y2=np.concatenate((Y,center), axis=0)
# mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
#                    dissimilarity="precomputed", n_jobs=1)
# D = metrics.euclidean_distances(Y2)
# mds.fit(D)
# mdsXY=mds.embedding_
# stress=mds.stress_

## PCA
pca = PCA(n_components=30)
pca.fit(Y)
Ypca=pca.transform(Y)
Cpca=pca.transform(center)

## rename clusters
ythe=np.array([0.5])
label,Cpca,center=SSRS.cluster_rename(label,ythe,Cpca,center)

## plot PCA and cluster after resign name
SSRS.cluster_plot(Y,label,center)
plt.savefig(figdir+'Cluster')

p1=plt.scatter(Ypca[:,0],Ypca[:,1],c=label)
p2=plt.plot(Cpca[:,0],Cpca[:,1],'rx',markersize=20,markeredgewidth=5)
plt.colorbar()
plt.savefig(figdir+'PCA')

## compute distance
dist=np.zeros([nind,nc])
distx=np.zeros([nind,nc])
disty=np.zeros([nind,nc])
for j in range(0,nind):
    for i in range(0,nc):
        dist[j,i]=scipy.spatial.distance.euclidean(Y[j,:],center[i,:])
        distx[j,i]=Ypca[j,0]-Cpca[i,0]
        disty[j,i]=Ypca[j,1]-Cpca[i,1]

## plot attr vs dist
figname=figdir+'\dist\dist_attr'
SSRS.DistPlot(Xn,dist,figname,field)
figname=figdir+'\dist_PCA1\distPCA1_attr'
SSRS.DistPlot(Xn,distx,figname,field)
figname=figdir+'\dist_PCA2\distPCA2_attr'
SSRS.DistPlot(Xn,disty,figname,field)

## regression
regModel=linear_model.LinearRegression()
#regModel=svm.SVC()
regModel=KNeighborsRegressor(n_neighbors=10)
regModel = tree.DecisionTreeRegressor()
regModel = GaussianNB()

Yp,Ytest,rmse_band=SSRS.Regression(Y=dist,X=Xn,prec=0.2,regModel=regModel,doplot=2)

#feature selection
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(\
        Xn,dist,test_size=0.2,random_state=0)
regModel=linear_model.LinearRegression()
attr_a0,score_a0,scoreRef_a0=SSRS.FeatureSelect_plot(X_train,Y_train,X_test,Y_test,regModel,opt=0,figname=figdir+"LR_F")
attr_a1,score_a1,scoreRef_a1=SSRS.FeatureSelect_plot(X_train,Y_train,X_test,Y_test,regModel,opt=1,figname=figdir+"LR_B")

regModel=KNeighborsRegressor(n_neighbors=10)
attr_b0,score_b0,scoreRef_b0=SSRS.FeatureSelect_plot(X_train,Y_train,X_test,Y_test,regModel,opt=0,figname=figdir+"KNN_F")
attr_b1,score_b1,scoreRef_b1=SSRS.FeatureSelect_plot(X_train,Y_train,X_test,Y_test,regModel,opt=1,figname=figdir+"KNN_B")

regModel = tree.DecisionTreeRegressor()
attr_c0,score_c0,scoreRef_a0=SSRS.FeatureSelect_plot(X_train,Y_train,X_test,Y_test,regModel,opt=0,figname=figdir+"Tree_F")
attr_c1,score_c1,scoreRef_c1=SSRS.FeatureSelect_plot(X_train,Y_train,X_test,Y_test,regModel,opt=1,figname=figdir+"Tree_B")

# write attrbute scores to excel
as_a0=np.argsort(attr_a0,axis=0)
as_a1=np.argsort(attr_a1,axis=0)[::-1]
as_b0=np.argsort(attr_b0,axis=0)
as_b1=np.argsort(attr_b1,axis=0)[::-1]
as_c0=np.argsort(attr_c0,axis=0)
as_c1=np.argsort(attr_c1,axis=0)[::-1]
attrscore=np.array([as_a0,as_a1,as_b0,as_b1,as_c0,as_c1])
attrscore=np.concatenate((as_a0,as_a1,as_b0,as_b1,as_c0,as_c1),axis=2)

attr_sel,score,scoreRef=SSRS.FeatureSelectForward\
    (X_train,Y_train[:,0],X_test,Y_test[:,0],regModel)
attr_rem,score,scoreRef=SSRS.FeatureSelectBackward\
    (X_train,Y_train[:,0],X_test,Y_test[:,0],regModel)

## save to matlab
matfile="Y:\\Kuai\\USGSCorr\\frompython.mat"
sio.savemat(matfile,{"label":label,"center":center,"dist":dist})

mat=sio.loadmat("Y:\\Kuai\\USGSCorr\\S_I2.mat")
SI=mat['S_I']