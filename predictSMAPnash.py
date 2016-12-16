import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import itertools
from time import clock

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import linear_model
from sklearn import cross_validation

import SSRS

################################################################
# LOAD DATA
################################################################
dataFile='Y:\Kuai\rnnSMAP\CART\data.mat'

mat = sio.loadmat(dataFile)
UCData=mat['usgsCorr']
mat = sio.loadmat(dataFile)
xData=mat['xMat']
yData=mat['yMat']


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


################################################################
# CLUSTER
################################################################
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

## rename clusters
ythe=np.array([0])
label,Cpca,center=SSRS.Cluster_rename(label,ythe,Cpca,center)

## plot PCA and cluster after resign name
SSRS.Cluster_plot(Y,label,center)

plt.figure()
p1=plt.scatter(Ypca[:,0],Ypca[:,1],c=label)
p2=plt.plot(Cpca[:,0],Cpca[:,1],'rx',markersize=20,markeredgewidth=5)
plt.colorbar()

## compute distance
dist=np.zeros([nind,nc])
distx=np.zeros([nind,nc])
disty=np.zeros([nind,nc])
for j in range(0,nind):
    for i in range(0,nc):
        dist[j,i]=SSRS.DistCal(Y[j,:],center[i,:])
        distx[j,i]=Ypca[j,0]-Cpca[i,0]
        disty[j,i]=Ypca[j,1]-Cpca[i,1]

distmatfile=UCdir+"py_dist_"+dataname+".mat"
pcamatfile=UCdir+"py_pca_"+dataname+".mat"
sio.savemat(distmatfile, {"label":label,"center":center,"dist":dist,"indvalid":indvalid})
sio.savemat(pcamatfile, {"Ypca":Ypca,"Cpca":Cpca,"indvalid":indvalid})


################################################################
# FEATURE SELECTION (Multi)
################################################################
#predFull=[2,4,8,11,20,22,27,29,30,31,32,41,44,45,46,49,50]
predFull=[2,4,8,11,20,22,41,44,45,46,49,50]
predName=[Field[i] for i in predFull]
predList=list(itertools.combinations(predFull,8))
n=predList.__len__()
score=np.ones(n)

st=clock()
for i in range(0,n):
    rmsetemp=np.zeros([10])
    for k in range(0,10):
        ind = range(0, nind)
        X_train, X_test, Y_train, Y_test, ind_train, ind_test = \
            cross_validation.train_test_split(Xn, dist, ind, test_size=0.2, random_state=k)

        regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20, min_samples_leaf=20)
        fitModel=linear_model.LinearRegression()
        predSel=predList[i]
        Yp,Yptrain,regTreeModel,fitModelList,predind=\
            SSRS.RegressionTree(X_train[:,predSel],X_test[:,predSel],Y_train,Y_test,regTreeModel,fitModel,predName,
                                doFitSelection=0,doMultiBand=1)
        rmsetemp[k],rmse_band=SSRS.RMSECal(Yp,Y_test)
    rmse=rmsetemp.mean()
    score[i]=rmse
    if i%100==0:
        print(i)
        et=clock()
        print(et-st)
        st=clock()
        print(np.min(score))

#scorematfile=UCdir+"py_predScore_multi_"+dataname+".mat"
scorematfile=UCdir+"py_predScore_multi_"+dataname+"_nosoil.mat"
sio.savemat(scorematfile,{"score":score,"predList":predList,})

################################################################
# CROSS VALIDATION (Multi)
################################################################
#scorematfile=UCdir+"py_predScore_multi_"+dataname+".mat"
scorematfile=UCdir+"py_predScore_multi_"+dataname+"_nosoil.mat"
mat = sio.loadmat(scorematfile)
predList=mat['predList']
score=mat['score'].flatten()

def testModel(predListTest):
    nmodel = predListTest.__len__()
    nit = 10
    # test 1: different size of training and test
    print('test1')
    testErr = np.ones([nmodel, 5, nit])
    testsize = [0.2, 0.4, 0.5, 0.6, 0.7]
    for k in range(0, nit):
        print(k)
        for j in range(0, nmodel):
            for i in range(0, testsize.__len__()):
                ind = range(0, nind)
                X_train, X_test, Y_train, Y_test, ind_train, ind_test = \
                    cross_validation.train_test_split(Xn, dist,ind,test_size=testsize[i], random_state=k)
                regTreeModel = tree.DecisionTreeRegressor(max_leaf_nodes=20, min_samples_leaf=20)
                fitModel = linear_model.LinearRegression()
                predSel = predListTest[j]
                predName = [Field[jj] for jj in predSel]
                Yp, Yptrain, regTreeModel, fitModelList, predind = \
                    SSRS.RegressionTree(X_train[:, predSel],X_test[:, predSel],Y_train,Y_test,regTreeModel,
                                        fitModel, predName,
                                        doFitSelection=0, doMultiBand=1)
                rmse, rmse_band = SSRS.RMSECal(Yp, Y_test)
                testErr[j, i, k] = rmse

    # test 2: use 1 HUC2 as test
    print('test2')
    testErr1_huc2_rt = np.ones([nmodel, 18])
    trainErr1_huc2_rt = np.ones([nmodel, 18])
    IDhucfile = UCdir + "IDhuc_" + dataname + ".mat"
    mat = sio.loadmat(IDhucfile)
    IDhuc = mat["IDhuc"]
    huc2 = IDhuc[indvalid, 1]
    for k in range(0, nit):
        print(k)
        for i in range(0, 18):
            ind = range(0, nind)
            X_train, X_test, Y_train, Y_test, ind_train, ind_test = \
                cross_validation.train_test_split(Xn,dist,ind,test_size=0.2,random_state=k)
            ind_test = np.where(huc2 == i + 1)[0]
            X_test = Xn[ind_test, :]
            Y_test = dist[ind_test, :]
            for j in range(0, nmodel):
                regTreeModel = tree.DecisionTreeRegressor(max_leaf_nodes=20, min_samples_leaf=20)
                fitModel = linear_model.LinearRegression()
                predSel = predListTest[j]
                predName = [Field[jj] for jj in predSel]
                Yp, Yptrain, regTreeModel, fitModelList, predind = \
                    SSRS.RegressionTree(X_train[:, predSel], X_test[:, predSel], Y_train, Y_test, regTreeModel,
                                        fitModel, predName,
                                        doFitSelection=0, doMultiBand=1)
                rmse, rmse_band = SSRS.RMSECal(Yptrain, Y_train)
                trainErr1_huc2_rt[j, i] = rmse
                rmse, rmse_band = SSRS.RMSECal(Yp, Y_test)
                testErr1_huc2_rt[j, i] = rmse

    # test 3: leave out 1 HUC2 one time
    print('test3')
    testErr1_huc2 = np.ones([nmodel, 18])
    trainErr1_huc2 = np.ones([nmodel, 18])
    IDhucfile = UCdir + "IDhuc_" + dataname + ".mat"
    mat = sio.loadmat(IDhucfile)
    IDhuc = mat["IDhuc"]
    huc2 = IDhuc[indvalid, 1]
    for i in range(0, 18):
        print(i)
        ind_test = np.where(huc2 == i + 1)[0]
        ind_train = np.where(huc2 != i + 1)[0]
        X_train = Xn[ind_train, :]
        X_test = Xn[ind_test, :]
        Y_train = dist[ind_train,:]
        Y_test = dist[ind_test,:]
        for j in range(0, nmodel):
            regTreeModel = tree.DecisionTreeRegressor(max_leaf_nodes=20, min_samples_leaf=20)
            fitModel = linear_model.LinearRegression()
            predSel = predListTest[j]
            predName = [Field[jj] for jj in predSel]
            Yp, Yptrain, regTreeModel, fitModelList, predind = \
                SSRS.RegressionTree(X_train[:, predSel], X_test[:, predSel], Y_train, Y_test, regTreeModel, fitModel,
                                    predName,
                                    doFitSelection=0, doMultiBand=1)
            rmse, rmse_band = SSRS.RMSECal(Yptrain, Y_train)
            trainErr1_huc2[j, i] = rmse
            rmse, rmse_band = SSRS.RMSECal(Yp, Y_test)
            testErr1_huc2[j, i] = rmse

    # test 4: leave out 2 HUC2 one time
    print('test4')
    testErr2_huc2 = np.ones([nmodel, 18 * 17])
    trainErr2_huc2 = np.ones([nmodel, 18 * 17])
    hucTab = np.ones([18 * 17, 2])
    IDhucfile = UCdir + "IDhuc_" + dataname + ".mat"
    mat = sio.loadmat(IDhucfile)
    IDhuc = mat["IDhuc"]
    huc2 = IDhuc[indvalid, 1]
    n = -1
    for i in range(0, 18):
        print(i)
        for j in range(0, 18):
            if i == j:
                continue
            n = n + 1
            hucTab[n, 0] = i
            hucTab[n, 1] = j
            ind_test = np.where((huc2 == i + 1) | (huc2 == j + 1))[0]
            ind_train = np.where((huc2 != i + 1) & (huc2 != j + 1))[0]
            X_train = Xn[ind_train, :]
            X_test = Xn[ind_test, :]
            Y_train = dist[ind_train, :]
            Y_test = dist[ind_test, :]
            for k in range(0, nmodel):
                regTreeModel = tree.DecisionTreeRegressor(max_leaf_nodes=20)
                fitModel = linear_model.LinearRegression()
                predSel = predListTest[k]
                predName = [Field[jj] for jj in predSel]
                Yp, Yptrain, regTreeModel, fitModelList, predind = \
                    SSRS.RegressionTree(X_train[:, predSel], X_test[:, predSel], Y_train, Y_test, regTreeModel,
                                        fitModel, predName,
                                        doFitSelection=0, doMultiBand=1)
                rmse, rmse_band = SSRS.RMSECal(Yptrain, Y_train)
                trainErr2_huc2[k, n] = rmse
                rmse, rmse_band = SSRS.RMSECal(Yp, Y_test)
                testErr2_huc2[k, n] = rmse
    return testErr, trainErr1_huc2, testErr1_huc2, \
           trainErr1_huc2_rt, testErr1_huc2_rt, \
           trainErr2_huc2, testErr2_huc2, hucTab


ss=np.sort(score)
ssInd=np.argsort(score)
predListTest=predList[ssInd[0:100],:].tolist()

testErr,trainErr1_huc2,testErr1_huc2,trainErr1_huc2_rt,testErr1_huc2_rt,\
trainErr2_huc2,testErr2_huc2,hucTab=testModel(predListTest)
#predErrmatfile=UCdir+"py_predCV_"+dataname+".mat"
predErrmatfile=UCdir+"py_predCV_"+dataname+"_nosoil.mat"
sio.savemat(predErrmatfile,{"predListTest":predListTest,"testErr":testErr,
                            "trainErr1_huc2":trainErr1_huc2,"testErr1_huc2":testErr1_huc2,
                            "trainErr1_huc2_rt":trainErr1_huc2_rt,"testErr1_huc2_rt":testErr1_huc2_rt,
                            "trainErr2_huc2":trainErr2_huc2,"testErr2_huc2":testErr2_huc2,
                            "hucTab":hucTab})

# test for several cases
predListTest=[]
predListTest.append([8,4,41,43,22,44,45,46,50])
predListTest.append([8,4,41,47,22,44,45,46,50])
predListTest.append([8,4,41,2,22,44,45,46,50])
testErr,trainErr1_huc2,testErr1_huc2,trainErr1_huc2_rt,testErr1_huc2_rt,\
trainErr2_huc2,testErr2_huc2,hucTab=testModel(predListTest)
#predErrmatfile=UCdir+"py_predCV_"+dataname+".mat"
predErrmatfile=UCdir+"py_predCV_"+dataname+"_nosoil2.mat"
sio.savemat(predErrmatfile,{"predListTest":predListTest,"testErr":testErr,
                            "trainErr1_huc2":trainErr1_huc2,"testErr1_huc2":testErr1_huc2,
                            "trainErr1_huc2_rt":trainErr1_huc2_rt,"testErr1_huc2_rt":testErr1_huc2_rt,
                            "trainErr2_huc2":trainErr2_huc2,"testErr2_huc2":testErr2_huc2,
                            "hucTab":hucTab})


################################################################
# REGRESSION (Multi)
################################################################

predListmatfile=UCdir+"py_predList_"+dataname+".mat"
mat = sio.loadmat(predListmatfile)
predListTest=mat['predListTest'].flatten().tolist()

ind=range(0,nind)
X_train,X_test,Y_train,Y_test,ind_train,ind_test = \
    cross_validation.train_test_split(Xn,dist,ind,test_size=0.2,random_state=0)

predSel=predListTest[101].flatten()  #mB
predName=[Field[i] for i in predSel]
regTreeModel=tree.DecisionTreeRegressor(max_leaf_nodes=20)
fitModel=linear_model.LinearRegression()
Yp,Yptrain,regTreeModel,fitModelList,predind=\
    SSRS.RegressionTree(X_train[:,predSel],X_test[:,predSel],Y_train,Y_test,regTreeModel,fitModel,predName,
                        doFitSelection=0,doMultiBand=0)
rmse,rmse_band=SSRS.RMSECal(Yptrain,Y_train)
print(rmse)
rmse,rmse_band=SSRS.RMSECal(Yp,Y_test)
print(rmse)
SSRS.Regression_plot(Yptrain,Y_train,doplot=2)
SSRS.Regression_plot(Yp,Y_test,doplot=2)
