
import scipy.io as sio
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from math import sqrt

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AffinityPropagation
# from sklearn.cluster import Birch

from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import SGDClassifier

from sklearn import manifold

from sklearn import preprocessing

import SSRS



### read data
UCdir = "Y:\Kuai\USGSCorr\\"
UCfile=UCdir+"usgsCorr2.mat"
Datafile=UCdir+"dataset2.mat"
mat = sio.loadmat(UCfile)
UCData=mat['Corr_maxmin']
mat = sio.loadmat(Datafile)
AttrData=mat['dataset']

## preprocessing
X=UCData
XX=AttrData[:,0:51]
XX[np.isnan(XX)]=0
scaler=preprocessing.StandardScaler().fit(XX)
XXn=scaler.fit_transform(XX)



def fitGage(ind,X,doplot=0):
    x = np.linspace(0, 1, 15)
    y1=X[ind,0:15]
    y2=X[ind,15:30]

    def func_line(x,b,c):
        return b*x+c
    pl1,cl1 = curve_fit(func_line, x, y1,maxfev=100000000)
    pl2,cl2 = curve_fit(func_line, x, y2, maxfev=100000000)
    vl1=func_line(x,pl1[0],pl1[1])
    vl2=func_line(x,pl2[0],pl2[1])

    def func_sin(x,a,b,c):
        return a*np.sin(b*x+c)
    ps1,cs1=curve_fit(func_sin,x,y1-vl1,maxfev=100000000)
    ps2,cs2=curve_fit(func_sin,x,y2-vl2,maxfev=100000000)
    vs1=func_sin(x,ps1[0],ps1[1],ps1[2])
    vs2=func_sin(x,ps2[0],ps2[1],ps2[2])

    v1=vs1+vl1
    v2=vs2+vl2
    rmse1=sqrt(metrics.mean_squared_error(y1,v1))
    rmse2=sqrt(metrics.mean_squared_error(y2,v2))

    if doplot==1:
        plt.plot((X[ind,0:15]),'-*b')
        plt.plot((X[ind,15:30]),'-*r')
        plt.plot(v1,'--b')
        plt.plot(v2,'--r')
        plt.title("ind=%s;rmse=%.3f,%.3f"%(ind,rmse1,rmse2))

    return pl1,pl2,ps1,ps2,rmse1,rmse2

ind=np.random.randint(0,X.shape[0]-1)
pl1,pl2,ps1,ps2,rmse1,rmse2=fitGage(ind,X,1)

X2=np.zeros((X.shape[0],10))
rmse=np.zeros((X.shape[0],2))
for i in range(0,X.shape[0]):
    print("%s"%i)
    pl1,pl2,ps1,ps2,rmse1,rmse2=fitGage(i,X)
    X2[i,]=np.concatenate((pl1,ps1,pl2,ps2))
    rmse[i,]=np.append(rmse1,rmse2)

np.savetxt("fitpar.csv", X2, delimiter=",")
np.savetxt("fitrmse.csv", rmse, delimiter=",")
plt.hist(X2[:,0],bins=10)
colors="bgrcmykw"
for i in range(0,10):
    h=np.histogram(X2[:,i],bins=10)
    plt.plot(h[0],colors[i])

f, axarr = plt.subplots(10,10)
for i in range(0,10):
    for j in range(0,10):
        axarr[i, j].plot(X2[:,i],X2[:,j],'.')


# preprocessing
scaler=preprocessing.MinMaxScaler()
X2n=scaler.fit_transform(X2)

#cluster
kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10,max_iter=1000)
kmeans.fit(X2n[:,0:5])
SSRS.clusterplot(X2n[:,0:5],label=kmeans.labels_,center=kmeans.cluster_centers_)

db = DBSCAN(eps=1, min_samples=10).fit(X2n)
nclass = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
SSRS.clusterplot(X2n,label=db.labels_+1)

af = AffinityPropagation(preference=-150,verbose=True)
af.fit(X2n)
SSRS.clusterplot(X2n,label=af.labels_)

#classification
T=kmeans.labels_
nfold=10
# Decision Tree
model=tree.DecisionTreeClassifier()
Tp=SSRS.ClusterLearn_cross(XXn,T,nfold,model)
SSRS.plotErrorMap(T,Tp)

#manifold
Y2 = manifold.Isomap(10, 2).fit_transform(X2n)
Y = manifold.Isomap(10, 2).fit_transform(X)
plt.plot(Y2[:,0],Y2[:,1],'*')

mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                   dissimilarity="precomputed", n_jobs=1)
D = metrics.euclidean_distances(X)
mds.fit(D)
Y=mds.embedding_
stress=mds.stress_

np.savetxt("fitMDS.csv",mds.embedding_, delimiter=",")