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

## read data
#from predictCorr import Yptrain

#UCdir = "E:\work\SSRS\data\\"
UCdir = "D:\Kuai\SSRS\data\\"
# UCdir = r"/Volumes/wrgroup/Kuai/USGSCorr/"
UCfile=UCdir+"usgsCorr_mB_4949.mat"
Datafile=UCdir+"dataset_mB_4949.mat"
# UCfile=UCdir+"usgsCorr_14_4881.mat"
# Datafile=UCdir+"dataset_14_4881.mat"
# UCfile=UCdir+"usgsCorr_12_4919.mat"
# Datafile=UCdir+"dataset_12_4919.mat"

mat = sio.loadmat(UCfile)
UCData=mat['usgsCorr']
mat = sio.loadmat(Datafile)
AttrData=mat['dataset']
Field = [str(''.join(letter)) for letter_array in mat['field'] for letter in letter_array]
figdir='Y:\\Kuai\\USGSCorr\\figures_dist\\'

## preprocessing
Field=[Field[i] for i in range(0,52)]
Y=UCData
# attrind=np.array(range(1,51)+range(62,78,3))
# Field=[Field[i] for i in range(1,51)+range(62,78,3)]
attrind=np.array(range(0,52))
X=AttrData[:,attrind]
#X[np.isnan(X)]=0
#Y[np.isnan(Y)]=0

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

# test for k in kmean
score_cluster=np.zeros(8)
for i in range(2,10):
    print(i)
    nc=i
    model = KMeans(init='k-means++', n_clusters=nc, n_init=10, max_iter=1000)
    label,center=SSRS.Cluster(Y, model,doplot=0)
    score_cluster[i-2]=metrics.silhouette_score(Y,label)
plt.plot(range(2,10),score_cluster,'-*')

## cluster
nc=6
model = KMeans(init='k-means++', n_clusters=nc, n_init=15, max_iter=1000,tol=1e-15,verbose=True)
label,center=SSRS.Cluster(Y, model,doplot=0)

## PCA
pca = PCA(n_components=nband)
pca.fit(Y)
Ypca=pca.transform(Y)
Cpca=pca.transform(center)
Ypca[:,0]=-Ypca[:,0]
Cpca[:,0]=-Cpca[:,0]
# Ypca[:,1]=-Ypca[:,1]
# Cpca[:,1]=-Cpca[:,1]

## rename clusters
ythe=np.array([0])
label,Cpca,center=SSRS.Cluster_rename(label,ythe,Cpca,center)

## plot PCA and cluster after resign name
SSRS.Cluster_plot(Y,label,center)
#plt.savefig(figdir+'Cluster')

plt.figure()
p1=plt.scatter(Ypca[:,0],Ypca[:,1],c=label)
p2=plt.plot(Cpca[:,0],Cpca[:,1],'rx',markersize=20,markeredgewidth=5)
plt.colorbar()
#plt.savefig(figdir+'PCA')

## compute distance
dist=np.zeros([nind,nc])
distx=np.zeros([nind,nc])
disty=np.zeros([nind,nc])
for j in range(0,nind):
    for i in range(0,nc):
        dist[j,i]=SSRS.DistCal(Y[j,:],center[i,:])
        distx[j,i]=Ypca[j,0]-Cpca[i,0]
        disty[j,i]=Ypca[j,1]-Cpca[i,1]

## save to matlab to plot in publishable format
distmatfile=r"E:\work\SSRS\data\py_dist_mB_4949.mat"
pcamatfile=r"E:\work\SSRS\data\py_pca_mB_4949.mat"
# distmatfile=r"E:\work\SSRS\data\py_dist_14_4881.mat"
# pcamatfile=r"E:\work\SSRS\data\py_pca_14_4881.mat"
# distmatfile=r"E:\work\SSRS\data\py_dist_12_4919.mat"
# pcamatfile=r"E:\work\SSRS\data\py_pca_12_4919.mat"

sio.savemat(distmatfile,{"label":label,"center":center,"dist":dist,"indvalid":indvalid})
sio.savemat(pcamatfile,{"Ypca":Ypca,"Cpca":Cpca,"indvalid":indvalid})

## plot attr vs dist
figname=figdir+'\dist\dist_attr'
SSRS.DistPlot(Xn,dist,figname,Field)
figname=figdir+'\dist_PCA1\distPCA1_attr'
SSRS.DistPlot(Xn,distx,figname,Field)
figname=figdir+'\dist_PCA2\distPCA2_attr'
SSRS.DistPlot(Xn,disty,figname,Field)

## regression
ind=range(0,nind)
X_train,X_test,Y_train,Y_test,ind_train,ind_test = \
    cross_validation.train_test_split(Xn,dist,ind,test_size=0.2,random_state=0)

XYmatfile=r"E:\work\SSRS\data\py_XY_mB_4949.mat"
sio.savemat(XYmatfile,{"X_train":X_train,"X_test":X_test,"Y_train":Y_train,"Y_test":Y_test})


###
regModel=linear_model.LinearRegression()
#regModel=svm.SVC()
regModel=KNeighborsRegressor(n_neighbors=10)
regModel = tree.DecisionTreeRegressor()
regModel = GaussianNB()
regModel=tree.DecisionTreeRegressor\
    (max_features=0.8,max_depth=10,min_samples_split=3,min_samples_leaf=5,
     min_weight_fraction_leaf=0.5,max_leaf_nodes=10)

Yp,Yptrain,regModelList=SSRS.Regression\
    (X_train,X_test,Y_train,Y_test,multiband=1,regModel=regModel,doplot=2)
rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
print(rmse)

## local regression tree
regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20)
fitModel=linear_model.LinearRegression()
Yp,Yptrain,regTreeModel,fitModelList,predind=\
    SSRS.RegressionTree(X_train,X_test,Y_train,Y_test,regTreeModel,fitModel,Field,doMultiBand=0)
rmse,rmse_band=SSRS.RMSECal(Yptrain,Y_train)
print(rmse)
rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
print(rmse)

SSRS.Regression_plot(Yptrain,Y_train,doplot=2)
SSRS.Regression_plot(Yp,Y_test,doplot=2)

## optimize by DE
from scipy.optimize import differential_evolution

def regtree(par,*data):
    print(par)
    X_train,X_test,Y_train,Y_test=data
    regTreeModel=tree.DecisionTreeRegressor\
        (max_features=par[0],min_samples_split=par[1],min_samples_leaf=par[2],
         min_weight_fraction_leaf=par[3],max_leaf_nodes=int(par[4]))
    fitModel=linear_model.LinearRegression()
    Yp,Yptrain,regTreeModel,fitModelList,predind=\
        SSRS.RegressionTree(X_train,X_test,Y_train,Y_test,regTreeModel,fitModel,Field,
                            doFitSelection=0,doMultiBand=1)
    rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
    print("RMSE=%f"%rmse)
    return rmse

data=(X_train,X_test,Y_train,Y_test)
bounds=[(0.001,0.999),(2,10),(2,100),(0,0.4999),(5,20)]
result = differential_evolution(regtree, bounds, args=data,tol=0.01,mutation=1.5,disp=True)


# 1. Predict dist band by band, 0.53394497629014659
# par=[  0.81712339,   2.64632913,  75.32099639,   0.31332268,  19.63184131]
# 2. Predict dist simultaneously, 0.52295255375604655
# par=[  0.49529808,   4.00610169,  47.56087887,   0.49946336,  19.39524644]
par=[  0.49529808,   4.00610169,  47.56087887,   0.49946336,  19.39524644]
regTreeModel=tree.DecisionTreeRegressor\
    (max_features=par[0],min_samples_split=par[1],min_samples_leaf=par[2],
     min_weight_fraction_leaf=par[3],max_leaf_nodes=int(par[4]))
fitModel=linear_model.LinearRegression()
Yp,Yptrain,regTreeModel,fitModelList,predind=\
    SSRS.RegressionTree(X_train,X_test,Y_train,Y_test,regTreeModel,fitModel,Field,
                        doFitSelection=0,doMultiBand=1)
rmse,rmse_band=SSRS.RMSECal(Yptrain,Y_train)
print(rmse)
rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
print(rmse)
SSRS.Regression_plot(Yptrain,Y_train,doplot=2)
SSRS.Regression_plot(Yp,Y_test,doplot=2)

# 3. select predictors
ind=range(0,nind)
X_train,X_test,Y_train,Y_test,ind_train,ind_test = \
    cross_validation.train_test_split(Xn,dist,ind,test_size=0.2,random_state=0)
# forward
regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20)
fitModel=linear_model.LinearRegression()
attr_sel,score,scoreRef=SSRS.FeatureSelectForward_RegTree\
    (X_train,X_test,Y_train,Y_test,regTreeModel,fitModel,Field,doFitSelection=0,doMultiBand=1)
attr_sel1,score1,scoreRef1=attr_sel,score,scoreRef
plt.figure()
plt.plot(score1,'-*b')
plt.plot(scoreRef1,'-*g')
#selmatfile=r"E:\work\SSRS\data\py_sel_14_4881.mat"
selmatfile=r"D:\Kuai\SSRS\data\py_selforward_mB_4949.mat"
sio.savemat(selmatfile,{"attr_sel":attr_sel,"score":score,"scoreRef":scoreRef})

#backward
regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20)
fitModel=linear_model.LinearRegression()
attr_sel,score,scoreRef=SSRS.FeatureSelectBackward_RegTree\
    (X_train,X_test,Y_train,Y_test,regTreeModel,fitModel,Field,doFitSelection=0,doMultiBand=1)
attr_sel2,score2,scoreRef2=attr_sel,score,scoreRef
plt.plot(score2,'-*b')
plt.plot(scoreRef2,'-*g')
selmatfile=r"D:\Kuai\SSRS\data\py_selbackward_mB_4949.mat"
sio.savemat(selmatfile,{"attr_sel":attr_sel,"score":score,"scoreRef":scoreRef})


nsel=8
predSel=attr_sel1[0:nsel]
predSel=[46, 11,8, 50, 41, 2, 22, 29]  #mB
# predSel=[45, 46, 9, 4, 40, 50, 25, 3] # 12
# predSel=[45, 46, 9, 4, 50, 26, 48, 31, 11, 2] #14
# predSel=[45, 46, 9, 4, 50, 26, 48, 31, 11, 2]
predName=[Field[i] for i in predSel]
regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20)
fitModel=linear_model.LinearRegression()
Yp,Yptrain,regTreeModel,fitModelList,predind=\
    SSRS.RegressionTree(X_train[:,predSel],X_test[:,predSel],Y_train,Y_test,regTreeModel,fitModel,predName,
                        doFitSelection=0,doMultiBand=1)
rmse,rmse_band=SSRS.RMSECal(Yptrain,Y_train)
print(rmse)
rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
print(rmse)
SSRS.Regression_plot(Yptrain,Y_train,doplot=2)
SSRS.Regression_plot(Yp,Y_test,doplot=2)

## save regression result to matlab
# regmatfile=r"E:\work\SSRS\data\py_reg_14_4881.mat"
regmatfile=r"E:\work\SSRS\data\py_reg_mB_4949.mat"
sio.savemat(regmatfile,{"Yp":Yp,"Yptrain":Yptrain,"Y_train":Y_train,"Y_test":Y_test,
                        "ind_train":ind_train,"ind_test":ind_test,"indvalid":indvalid})

## save traversed tree to matlab
regTree=regTreeModel.tree_
nnode=regTree.node_count
the=np.zeros([nnode])
fieldInd=regTree.feature
nodeValue=regTree.value
for i in range(0,nnode):
    if regTree.feature[i]!=-2:
        v=regTree.threshold[i]
        tempi=predSel[regTree.feature[i]]
        tempj=np.argmin(np.abs(Xn[:,tempi]-v))
        vout=X[tempj,tempi]
        the[i]=vout
        print("node %i, field %s, v=%f"%(i,Field[tempi],vout))
# treematfile=r"E:\work\SSRS\data\py_tree_14_4881.mat"
string,nodeind,leaf,label=SSRS.TraverseTree(regTree,Xn,predName)
treematfile=r"E:\work\SSRS\data\py_tree_mB_4949.mat"
sio.savemat(treematfile,{"string":string,"nodeind":nodeind,"leaf":leaf,
                         "label":label,"indvalid":indvalid,"the":the,
                         "fieldInd":fieldInd,"nodeValue":nodeValue,})
string,nodeind,leaf,label=SSRS.TraverseTree(regTree,X_train[:,predSel],predName)
# treematfile=r"E:\work\SSRS\data\py_tree_train_14_4881.mat"
treematfile=r"E:\work\SSRS\data\py_tree_train_mB_4949.mat"
sio.savemat(treematfile,{"string":string,"nodeind":nodeind,"leaf":leaf,
                         "label":label,"indvalid":indvalid,"ind_train":ind_train})
string,nodeind,leaf,label=SSRS.TraverseTree(regTree,X_test[:,predSel],predName)
# treematfile=r"E:\work\SSRS\data\py_tree_test_14_4881.mat"
treematfile=r"E:\work\SSRS\data\py_tree_test_mB_4949.mat"
sio.savemat(treematfile,{"string":string,"nodeind":nodeind,"leaf":leaf,
                         "label":label,"indvalid":indvalid,"ind_test":ind_test})


## plot tree
from sklearn import tree
treedot=r"E:\work\SSRS\data\tree_mB_4949.dot"
with open(treedot, 'w') as f:
    f = tree.export_graphviz(regTreeModel, out_file=f,feature_names=predName,
                             label='none',node_ids=True)


savedir=r"Y:\Kuai\SSRS\paper\12\tree\\"
SSRS.TreePlot(savedir,X_train[:,attr_sel1[0:nsel]],X_test[:,attr_sel1[0:nsel]],\
              Y_train,Y_test,regTreeModel,predName,fitModelList,predind)





#backward
regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20)
fitModel=linear_model.LinearRegression()
attr_sel,score,scoreRef=SSRS.FeatureSelectBackward_RegTree\
    (X_train,X_test,Y_train,Y_test,regTreeModel,fitModel,Field,doFitSelection=0,doMultiBand=1)
attr_sel2,score2,scoreRef2=attr_sel,score,scoreRef
plt.plot(score2,'-*b')
plt.plot(scoreRef2,'-*g')

nsel=12
Yp,Yptrain,regTreeModel,fitModelList,predind=\
    SSRS.RegressionTree(X_train[:,attr_sel2[-nsel-1:-1]],X_test[:,attr_sel2[-nsel-1:-1]],Y_train,Y_test,regTreeModel,fitModel,Field,
                        doFitSelection=0,doMultiBand=1)
rmse,rmse_band=SSRS.RMSECal(Yptrain,Y_train)
print(rmse)
rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
print(rmse)
predName=[Field[i] for i in attr_sel[-1:-nsel:-1]]

## after selection
selpred=[4,  9, 10, 22, 30, 39, 40, 43, 44, 45, 46]
regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20)
fitModel=linear_model.LinearRegression()
Yp,Yptrain,regTreeModel,fitModelList,predind=\
    SSRS.RegressionTree(X_train[:,selpred],X_test[:,selpred],Y_train,Y_test,regTreeModel,fitModel,Field,
                        doFitSelection=0,doMultiBand=1)
rmse,rmse_band=SSRS.RMSECal(Yptrain,Y_train)
print(rmse)
rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
print(rmse)

savedir=r"Y:\Kuai\USGSCorr\figure_tree_linreg\\"
SSRS.TreePlot(savedir,X_train,X_test,Y_train,Y_test,regTreeModel,Field,fitModelList,predind)



# full selection from
predFull=[2,  4,  8,  9, 11, 20, 22, 27, 29, 31, 41, 44, 45, 46, 49, 50]
predName=[Field[i] for i in predFull]
predList=list(itertools.combinations(predFull,8))
n=predList.__len__()
score=np.ones(n)

st=clock()
for i in range(0,n):
    regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20)
    fitModel=linear_model.LinearRegression()
    predSel=predList[i]
    Yp,Yptrain,regTreeModel,fitModelList,predind=\
        SSRS.RegressionTree(X_train[:,predSel],X_test[:,predSel],Y_train,Y_test,regTreeModel,fitModel,predName,
                            doFitSelection=0,doMultiBand=1)
    rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
    score[i]=rmse
    if i%100==0:
        print(i)
        et=clock()
        print(et-st)
        st=clock()
        print(np.min(score))
ss=np.sort(score)
ssInd=np.argsort(score)
np.sort(predList[ssInd[0]])
for i in range(0,50):
    print(np.sort(predList[ssInd[i]]))
    #print(score[ssInd[i]])
scorematfile=r"E:\work\SSRS\data\py_predScore_mB_4949.mat"
sio.savemat(scorematfile,{"score":score,"predList":predList,})


# test of selections
predFull=[2,  4,  8,  9, 11, 20, 22, 27, 29, 31, 41, 44, 45, 46, 49, 50]
predName=[Field[i] for i in predFull]
scorematfile=r"E:\work\SSRS\data\py_predScore_mB_4949.mat"
mat = sio.loadmat(scorematfile)
score=mat['score']
score=score.reshape(score.shape[1])
predList=mat['predList']
ss=np.sort(score)
ssInd=np.argsort(score)
predListTest=predList[ssInd[0:100]]


def testModel(predListTest):
    nmodel=predListTest.__len__()
    nit=50
    # test 1: different size of training and test
    testErr=np.ones([nmodel,5,50])
    testsize=[0.2,0.4,0.5,0.6,0.7]
    for k in range(0,nit):
        print(k)
        for j in range(0,nmodel):
            for i in range(0,5):
                ind=range(0,nind)
                X_train,X_test,Y_train,Y_test,ind_train,ind_test = \
                    cross_validation.train_test_split(Xn,dist,ind,test_size=testsize[i],random_state=k)
                regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20,min_samples_leaf=20)
                fitModel=linear_model.LinearRegression()
                predSel=predListTest[j]
                predName=[Field[jj] for jj in predSel]
                Yp,Yptrain,regTreeModel,fitModelList,predind=\
                    SSRS.RegressionTree(X_train[:,predSel],X_test[:,predSel],Y_train,Y_test,regTreeModel,fitModel,predName,
                                        doFitSelection=0,doMultiBand=1)
                rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
                testErr[j,i,k]=rmse

    # test 2: use 1 HUC2 as test
    testErr1_huc2_rt=np.ones([nmodel,18])
    trainErr1_huc2_rt=np.ones([nmodel,18])
    IDhucfile=r"E:\work\SSRS\data\IDhuc_mb_4949.mat"
    mat = sio.loadmat(IDhucfile)
    IDhuc=mat["IDhuc"]
    huc2=IDhuc[indvalid,1]
    for k in range(0,nit):
        print(k)
        for i in range(0,18):
            ind=range(0,nind)
            X_train,X_test,Y_train,Y_test,ind_train,ind_test = \
                cross_validation.train_test_split(Xn,dist,ind,test_size=0.2,random_state=k)
            ind_test=np.where(huc2==i+1)[0]
            X_test=Xn[ind_test,:]
            Y_test=dist[ind_test,:]
            for j in range(0,nmodel):
                regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20)
                fitModel=linear_model.LinearRegression()
                predSel=predListTest[j]
                predName=[Field[jj] for jj in predSel]
                Yp,Yptrain,regTreeModel,fitModelList,predind=\
                    SSRS.RegressionTree(X_train[:,predSel],X_test[:,predSel],Y_train,Y_test,regTreeModel,fitModel,predName,
                                        doFitSelection=0,doMultiBand=1)
                rmse,rmse_band=SSRS.RMSECal(Yptrain,Y_train)
                trainErr1_huc2_rt[j,i]=rmse
                rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
                testErr1_huc2_rt[j,i]=rmse

    # test 3: leave out 1 HUC2 one time
    testErr1_huc2=np.ones([nmodel,18])
    trainErr1_huc2=np.ones([nmodel,18])
    IDhucfile=r"E:\work\SSRS\data\IDhuc_mb_4949.mat"
    mat = sio.loadmat(IDhucfile)
    IDhuc=mat["IDhuc"]
    huc2=IDhuc[indvalid,1]
    for i in range(0,18):
        ind_test=np.where(huc2==i+1)[0]
        ind_train=np.where(huc2!=i+1)[0]
        X_train=Xn[ind_train,:]
        X_test=Xn[ind_test,:]
        Y_train=dist[ind_train,:]
        Y_test=dist[ind_test,:]
        for j in range(0,nmodel):
            regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20)
            fitModel=linear_model.LinearRegression()
            predSel=predListTest[j]
            predName=[Field[jj] for jj in predSel]
            Yp,Yptrain,regTreeModel,fitModelList,predind=\
                SSRS.RegressionTree(X_train[:,predSel],X_test[:,predSel],Y_train,Y_test,regTreeModel,fitModel,predName,
                                    doFitSelection=0,doMultiBand=1)
            rmse,rmse_band=SSRS.RMSECal(Yptrain,Y_train)
            trainErr1_huc2[j,i]=rmse
            rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
            testErr1_huc2[j,i]=rmse

    # test 4: leave out 2 HUC2 one time
    testErr2_huc2=np.ones([nmodel,18*17])
    trainErr2_huc2=np.ones([nmodel,18*17])
    hucTab=np.ones([18*17,2])
    IDhucfile=r"E:\work\SSRS\data\IDhuc_mb_4949.mat"
    mat = sio.loadmat(IDhucfile)
    IDhuc=mat["IDhuc"]
    huc2=IDhuc[indvalid,1]
    n=-1
    for i in range(0,18):
        print(i)
        for j in range(0,18):
            if i==j:
                continue
            n=n+1
            hucTab[n,0]=i
            hucTab[n,1]=j
            ind_test=np.where((huc2==i+1) | (huc2==j+1))[0]
            ind_train=np.where((huc2!=i+1) & (huc2!=j+1))[0]
            X_train=Xn[ind_train,:]
            X_test=Xn[ind_test,:]
            Y_train=dist[ind_train,:]
            Y_test=dist[ind_test,:]
            for k in range(0,nmodel):
                regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20)
                fitModel=linear_model.LinearRegression()
                predSel=predListTest[k]
                predName=[Field[jj] for jj in predSel]
                Yp,Yptrain,regTreeModel,fitModelList,predind=\
                    SSRS.RegressionTree(X_train[:,predSel],X_test[:,predSel],Y_train,Y_test,regTreeModel,fitModel,predName,
                                        doFitSelection=0,doMultiBand=1)
                rmse,rmse_band=SSRS.RMSECal(Yptrain,Y_train)
                trainErr2_huc2[k,n]=rmse
                rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
                testErr2_huc2[k,n]=rmse
    return testErr,trainErr1_huc2,testErr1_huc2,\
           trainErr1_huc2_rt,testErr1_huc2_rt,\
           trainErr2_huc2,testErr2_huc2,hucTab

predListTest=predList[ssInd[0:100]]
predListTest=predListTest.tolist()
predListTest.append([4,9,22,31,41,45,50])
predListTest.append([2,4,9,22,31,45,50])
predListTest.append([4,9,22,29,31,45,50])
predListTest.append([4,9,22,31,45,50])
predListTest.append([2,4,9,22,29,31,41,45,50])

testErr,trainErr1_huc2,testErr1_huc2,trainErr1_huc2_rt,testErr1_huc2_rt,\
trainErr2_huc2,testErr2_huc2,hucTab=testModel(predListTest)

predErrmatfile=r"E:\work\SSRS\data\py_predErr2_mB_4949.mat"
sio.savemat(predErrmatfile,{"predListTest":predListTest,"testErr":testErr,
                            "trainErr1_huc2":trainErr1_huc2,"testErr1_huc2":testErr1_huc2,
                            "trainErr1_huc2_rt":trainErr1_huc2_rt,"testErr1_huc2_rt":testErr1_huc2_rt,
                            "trainErr2_huc2":trainErr2_huc2,"testErr2_huc2":testErr2_huc2,
                            "hucTab":hucTab})
predListmatfile=r"E:\work\SSRS\data\py_predList_mB_4949.mat"
sio.savemat(predListmatfile,{"predListTest":predListTest})

# test 4: try specific cases
savedir=r"D:\Kuai\SSRS\tree\\"
nmList=[15,53,20,73,102]
for nm in nmList:
    predSel=predListTest[nm-1]
    predName=[Field[i] for i in predSel]
    print(predName)
    nit=50
    temp=np.ones([nit])
    treelist=[]
    trainIndList=[]
    testIndList=[]
    for k in range(0,nit):
        ind=range(0,nind)
        X_train,X_test,Y_train,Y_test,ind_train,ind_test = \
            cross_validation.train_test_split(Xn,dist,ind,test_size=0.2,random_state=k)
        regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20,min_samples_leaf=20)
        fitModel=linear_model.LinearRegression()
        Yp,Yptrain,regTreeModel,fitModelList,predind=\
            SSRS.RegressionTree(X_train[:,predSel],X_test[:,predSel],Y_train,Y_test,regTreeModel,fitModel,predName,
                                doFitSelection=0,doMultiBand=1)
        rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
        temp[k]=rmse
        treelist.append(regTreeModel)
        trainIndList.append(ind_train)
        testIndList.append(ind_test)
    print(temp.mean())
    print(temp.std())
    tempSort=np.sort(temp)
    tempSortInd=np.argsort(temp)

    treefolder=r"Y:\Kuai\SSRS\trees"
    ntop=5
    for k in range(0,ntop):
        treename="tree#%s_%s"%(nm,k)
        kk=tempSortInd[k]
        X_train,X_test,Y_train,Y_test,ind_train,ind_test = \
            cross_validation.train_test_split(Xn,dist,ind,test_size=0.2,random_state=kk)
        regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20,min_samples_leaf=20)
        fitModel=linear_model.LinearRegression()
        Yp,Yptrain,regTreeModel,fitModelList,predind=\
            SSRS.RegressionTree(X_train[:,predSel],X_test[:,predSel],Y_train,Y_test,regTreeModel,fitModel,predName,
                                doFitSelection=0,doMultiBand=1)
        rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
        with open(savedir+"\\"+treename+".dot", 'w') as f:
            f = tree.export_graphviz(regTreeModel, out_file=f,feature_names=predName,
                                     label='none',node_ids=True)
        regTree=regTreeModel.tree_
        nnode=regTree.node_count
        the=np.zeros([nnode])
        fieldInd=regTree.feature
        nodeValue=regTree.value
        for i in range(0,nnode):
            if regTree.feature[i]!=-2:
                v=regTree.threshold[i]
                tempi=predSel[regTree.feature[i]]
                tempj=np.argmin(np.abs(Xn[:,tempi]-v))
                vout=X[tempj,tempi]
                the[i]=vout
                print("node %i, field %s, v=%f"%(i,Field[tempi],vout))
        string,nodeind,leaf,label=SSRS.TraverseTree(regTree,Xn,predName)
        treematfile=treefolder+"\\"+treename+".mat"
        sio.savemat(treematfile,{"string":string,"nodeind":nodeind,"leaf":leaf,
                                 "label":label,"indvalid":indvalid,"the":the,
                                 "cleft":regTree.children_left,"cright":regTree.children_right,
                                 "fieldInd":fieldInd,"nodeValue":nodeValue,})
        string,nodeind,leaf,label=SSRS.TraverseTree(regTree,X_train[:,predSel],predName)
        # treematfile=r"E:\work\SSRS\data\py_tree_train_14_4881.mat"
        treematfile=treefolder+"\\"+treename+"_train.mat"
        sio.savemat(treematfile,{"string":string,"nodeind":nodeind,"leaf":leaf,
                                 "label":label,"indvalid":indvalid,"ind_train":ind_train})
        string,nodeind,leaf,label=SSRS.TraverseTree(regTree,X_test[:,predSel],predName)
        # treematfile=r"E:\work\SSRS\data\py_tree_test_14_4881.mat"
        treematfile=treefolder+"\\"+treename+"_test.mat"
        sio.savemat(treematfile,{"string":string,"nodeind":nodeind,"leaf":leaf,
                                 "label":label,"indvalid":indvalid,"ind_test":ind_test})
        regmatfile=treefolder+"\\"+treename+"_reg.mat"
        sio.savemat(regmatfile,{"Yp":Yp,"Yptrain":Yptrain,"Y_train":Y_train,"Y_test":Y_test,
                                "ind_train":ind_train,"ind_test":ind_test,"indvalid":indvalid})


predSel=[2,4,9,22,31,45,49,50]
predName=[Field[i] for i in predSel]
X_train,X_test,Y_train,Y_test,ind_train,ind_test = \
    cross_validation.train_test_split(Xn,dist,ind,test_size=0.2)
regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20)
fitModel=linear_model.LinearRegression()
Yp,Yptrain,regTreeModel,fitModelList,predind=\
    SSRS.RegressionTree(X_train[:,predSel],X_test[:,predSel],Y_train,Y_test,regTreeModel,fitModel,predName,
                        doFitSelection=0,doMultiBand=1)
rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
print(rmse)
SSRS.Regression_plot(Yptrain,Y_train,doplot=2)
SSRS.Regression_plot(Yp,Y_test,doplot=2)



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

