import scipy.io as sio
import scipy
import matplotlib.pyplot as plt
import numpy as np

from sklearn import preprocessing
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA

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
