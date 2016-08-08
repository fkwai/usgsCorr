
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
UCdir =r"E:\work\SSRS\\"
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

# select model
regModel=linear_model.LinearRegression()
#regModel=svm.SVC()
regModel=KNeighborsRegressor(n_neighbors=20)
regModel=tree.DecisionTreeRegressor()
regModel=GaussianNB()
regModel=sklearn.linear_model.SGDRegressor()
regModel=RandomForestRegressor()

#regModel = BernoulliRBM(random_state=0, verbose=True)

# nn = Regressor(
#     layers=[
#         Layer("Sigmoid", units=200),
#         Layer("Sigmoid", units=200),
#         Layer("Linear")],
#     learning_rate=0.1,
#     n_iter=200,verbose=1)

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(\
        Xn,Y,test_size=0.2,random_state=0)

# predict correlation
Yp,Yptrain,regModelList=SSRS.Regression\
    (X_train,X_test,Y_train,Y_test,multiband=1,regModel=regModel,doplot=0)
rmse,rmse_band=SSRS.RMSEcal(Yp,Y_test)
rmse_train,rmse_band_train=SSRS.RMSEcal(Yptrain,Y_train)
print(rmse)
print(rmse_train)
print(np.corrcoef(Yp[:,0],Y_test[:,0]))

par=[1.0,16,6,20,0.15]
regTreeModel=tree.DecisionTreeRegressor\
    (max_features=par[0],max_depth=par[1],min_samples_split=par[2],min_samples_leaf=par[3],
     min_weight_fraction_leaf=par[4],max_leaf_nodes=18)
fitModel=linear_model.LinearRegression()
Yp,Yptrain,regTreeModel,fitModelList,predind=SSRS.RegressionTree\
    (X_train,X_test,Y_train,Y_test,regTreeModel,fitModel,Field,doFitSelection=0)
rmse,rmse_band=SSRS.RMSEcal(Yp,Y_test)
rmse_train,rmse_band_train=SSRS.RMSEcal(Yptrain,Y_train)
print(rmse)
print(rmse_train)


# predict correlation
regModel=tree.DecisionTreeRegressor\
    (max_features=0.3,max_depth=20,min_samples_split=3,min_samples_leaf=5,
     min_weight_fraction_leaf=0.5)
regModel=GradientBoostingRegressor()
Yp,rmse,rmse_train,rmse_band,rmse_band_train=SSRS.Regression\
    (X_train,X_test,Y_train,Y_test,multiband=0,regModel=regModel,doplot=0)
print(rmse)
print(rmse_train)

regModel.fit(X_train, Y_train)
savedir=r"/Volumes/wrgroup/Kuai/USGSCorr/figure_tree/"
savedir=r"Y:\Kuai\USGSCorr\figure_tree\\"
with open(savedir+"tree.dot", 'w') as f:
    f = tree.export_graphviz(regModel, out_file=f,feature_names=Field,
                             label='none',node_ids=True)
os.system("dot -Tpng tree.dot -o tree.png")

# optimize parameters
from scipy.optimize import differential_evolution

def regtree(par,*data):
    X_train,X_test,Y_train,Y_test=data
    regTreeModel=tree.DecisionTreeRegressor\
        (max_features=par[0],min_samples_split=par[1],min_samples_leaf=par[2],
         min_weight_fraction_leaf=par[3],max_leaf_nodes=int(par[4]))
    fitModel=linear_model.LinearRegression()
    Yp,Yptrain,regTreeModel,fitModelList,predind=\
        SSRS.RegressionTree(X_train,X_test,Y_train,Y_test,regTreeModel,fitModel,Field,
                            doFitSelection=0,doMultiBand=1)
    rmse,rmse_band=SSRS.RMSEcal(Yp,Y_test)
    print(rmse)
    return rmse

data=(X_train,X_test,Y_train,Y_test)
bounds=[(0.001,0.999),(2,10),(2,100),(0,0.4999),(5,20)]
result = differential_evolution(regtree, bounds, args=data,tol=0.01,mutation=1.5)

# 1. Predict corr band by band, 0.21991421646783735
# par=[  0.39304037,   7.54346815,  93.41259044,   0.32280695,  18.76335834]
# 2. Predict dist simultaneously, 0.52295255375604655
# par=[  1.10901464e-01,4.16068961e+00,3.65341741e+01,1.34070518e-02,1.96304261e+01]
par=[  0.39304037,   7.54346815,  93.41259044,   0.32280695,  18.76335834]
par=[  1.10901464e-01,4.16068961e+00,3.65341741e+01,1.34070518e-02,1.96304261e+01]

regTreeModel=tree.DecisionTreeRegressor\
    (max_features=par[0],min_samples_split=par[1],min_samples_leaf=par[2],
     min_weight_fraction_leaf=par[3],max_leaf_nodes=int(par[4]))
fitModel=linear_model.LinearRegression()
Yp,Yptrain,regTreeModel,fitModelList,predind=\
    SSRS.RegressionTree(X_train,X_test,Y_train,Y_test,regTreeModel,fitModel,Field,
                        doFitSelection=0,doMultiBand=1)
rmse,rmse_band=SSRS.RMSEcal(Yp,Y_test)
rmse,rmse_band=SSRS.RMSEcal(Yptrain,Y_train)
print(rmse)
rmse,rmse_band=SSRS.RMSEcal(Yp,Y_test)
print(rmse)
SSRS.Regression_plot(Yptrain,Y_train,doplot=1)
SSRS.Regression_plot(Yp,Y_test,doplot=1)


## select attributes
par=[1.0,16,6,20,0.15]
regModel=tree.DecisionTreeRegressor\
    (max_features=par[0],max_depth=par[1],min_samples_split=par[2],min_samples_leaf=par[3],
     min_weight_fraction_leaf=par[4],max_leaf_nodes=10)
attr_rem,score,scoreRef=SSRS.FeatureSelectBackward\
    (X_train,Y_train,X_test,Y_test,regModel)
attr_sel,score,scoreRef=SSRS.FeatureSelectForward\
    (X_train,Y_train,X_test,Y_test,regModel)

nf=8
attrsel=np.array(list(reversed(attr_rem))[0:nf])
attrsel=np.array(attr_sel[0:nf])

#attrind=np.array([44, 45, 49, 40, 19, 25, 55])
#attrind=np.array(attr_sel[0:nf])

X2=AttrData[:,attrsel]
X2[np.isnan(X2)]=0
scaler=preprocessing.StandardScaler().fit(X2)
Xn2=scaler.fit_transform(X2)

X2_train,X2_test,Y2_train,Y2_test = cross_validation.train_test_split(\
        Xn2,Y,test_size=0.2,random_state=0)

Yp,rmse,rmse_train,rmse_band,rmse_band_train=SSRS.Regression\
    (X2_train,X2_test,Y2_train,Y2_test,multiband=1,regModel=regModel,doplot=0)
print(attrind.shape[0])
print(rmse)
print(rmse_train)
print([Field[i] for i in attrind])

## plot tree
regModel.fit(X2_train, Y2_train)
savedir=r"/Volumes/wrgroup/Kuai/USGSCorr/figure_tree/"
savedir=r"Y:\Kuai\USGSCorr\figure_tree\\"

with open(savedir+"tree.dot", 'w') as f:
    f = tree.export_graphviz(regModel, out_file=f,feature_names=[Field[i] for i in attrsel],
                             label='none',node_ids=True)
os.system("dot -Tpng tree.dot -o tree.png")

regTree=regModel.tree_
feature_names=[Field[i] for i in attrind]
Xin=X2_train
Yin=Y2_train
string,nodeind,leaf,label=SSRS.traverseTree(regTree,feature_names,Xin)
for i in range(0,regTree.node_count):
    plt.figure()
    plt.boxplot(Yin[nodeind[i],:])
    plt.title(string[i],fontsize=8)
    #plt.tight_layout()
    plt.savefig(savedir+"Train_node%i"%i)
    plt.close()

Xin=X2_test
Yin=Y2_test
string,nodeind,leaf,label=SSRS.traverseTree(regTree,feature_names,Xin)
for i in range(0,regTree.node_count):
    plt.figure()
    plt.boxplot(Yin[nodeind[i],:])
    plt.title(string[i],fontsize=8)
    #plt.tight_layout()
    plt.savefig(savedir+"Test_node%i"%i)
    plt.close()

## PCA of tree leaves
## PCA
Xin=Xn2
Yin=Y
string,nodeind,leaf,label=SSRS.traverseTree(regTree,feature_names,Xin)

nclass=np.unique(leaf).shape[0]
nband=30
pca = PCA(n_components=30)
pca.fit(Yin)
Ypca=pca.transform(Yin)
center=np.zeros([nclass,nband])
for i in range(0,nclass):
    center[i,:]=Yin[nodeind[i],:].mean(axis=0)
Cpca=pca.transform(center)

p1=plt.scatter(Ypca[:,0],Ypca[:,1],c=label)
p2=plt.plot(Cpca[:,0],Cpca[:,1],'rx',markersize=20,markeredgewidth=5)
plt.colorbar()

SSRS.Cluster_plot(Y,label.astype(int))
