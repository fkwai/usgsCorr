
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy
from sklearn import metrics
from sklearn import cross_validation
from matplotlib import gridspec
import itertools
from time import clock
import scipy.io as sio

import plotFun


def Cluster(data,model,doplot=1):
    model.fit(data)
    label=model.labels_
    if hasattr(model, 'cluster_centers_'):
        center=model.cluster_centers_
    else:
        center=[]
    if doplot==1:
        Cluster_plot(data,label,center)
    return label,center

def Cluster_plot(data,label,center=[],rank=0):
    labeluniq=np.unique(label)
    nclass=len(labeluniq)
    nind_class=np.bincount(label)[labeluniq]
    if rank==1: # rank subfigures based on # of ind of each clusters
        sortind=np.argsort(nind_class)[::-1]
    else:
        sortind=range(0,nclass)
    score=metrics.silhouette_score(data,label)

    if nclass>10:
        n=10
    else:
        n=nclass
    f, axarr = plt.subplots(int(np.ceil(n/2.)), 2)
    #f.tight_layout()
    f.subplots_adjust(top=0.9)
    plt.suptitle('Silhouette Coefficient=%s'%score)
    for i in range(n):
        ind=sortind[i]
        pj=int(np.ceil(i/2))
        pi=int(i%2)
        axarr[pj, pi].boxplot(data[label==labeluniq[ind]])
        axarr[pj, pi].set_title('cluster %s, nind=%s'%(labeluniq[ind],nind_class[ind]))
        axarr[pj, pi].set_ylim([-1,1])
        if len(center)>0:
            axarr[pj, pi].plot(range(1,31),center[ind,],'*-r')

def Cluster_rename(label,ythe,Cpca,center=[]):
    #rename Clusters based on 2 PCA of centers.
    #ythe=np.array([0.5])
    lab0=label
    nind=len(label)
    nc=len(np.unique(label))

    lab1=label*0-1
    labind1=np.zeros([nc])-1
    Cpcanew=Cpca*0;
    centernew=center*0

    #figure out labind1
    n=0
    for i in range(0,len(ythe)):
        if i==0:
            ind1=np.where(Cpca[:,1]<ythe[i])[0]
        else:
            ind1=np.where(Cpca[:,1]<ythe[i]&Cpca[:,1]>ythe[i-1])[0]
        n1=len(ind1)
        x1=Cpca[ind1,0]
        indtemp=ind1[np.argsort(x1)]
        labind1[indtemp]=range(n,n+n1)
        n=n+n1
    ind2=np.where(Cpca[:,1]>ythe[len(ythe)-1])[0]
    n2=len(ind2)
    x2=Cpca[ind2,0]
    indtemp=ind2[np.argsort(x2)]
    labind1[indtemp]=range(n,n+n2)
    n=n+n2
    #resign label
    for i in range(0,nc):
        ind=np.where(lab0==i)
        labnew=labind1[i]
        lab1[ind]=labnew
        Cpcanew[labnew,:]=Cpca[i,:]
        if len(center)>0:
            centernew[labnew,:]=center[i,:]

    return lab1,Cpcanew,centernew

def Classification_cross(XXn,T,nfold,model):
    indran=range(len(T))
    random.shuffle(indran)
    indran=np.asarray(indran)
    ngroup=int(np.floor(len(T)/nfold))
    Tp=np.zeros((len(T),),int)

    for i in range(10):
        if i==nfold-1:
            ind=indran[range(i*ngroup,len(indran),1)]
        else:
            ind=indran[range(i*ngroup,(i+1)*ngroup-1,1)]
        X1=XXn[ind]
        T1=T[ind]
        X0=np.delete(XXn,ind,0)
        T0=np.delete(T,ind)
        model = model.fit(X0, T0)
        Tp[ind]=model.predict(X1)
    return Tp

def plotErrorMap(T,Tp):
    #error map and performance plot
    nclass=len(np.unique(T))
    errormap=np.zeros((nclass,nclass),int)
    for i in range(len(T)):
        errormap[Tp[i],T[i]]=errormap[Tp[i],T[i]]+1

    plt.matshow(errormap)
    plt.ylabel("Predict")
    plt.xlabel("Truth")
    for i in range(nclass):
        for j in range(nclass):
            plt.text(j,i,errormap[j,i], va='center', ha='center',size=20)

    perf1=np.zeros((nclass),float)
    perf2=np.zeros((nclass),float)
    for i in range(nclass):
        perf1[i]=float(errormap[i,i])/float(np.sum(errormap[i,:]))
        perf2[i]=float(errormap[i,i])/float(np.sum(errormap[:,i]))
    plt.yticks(range(nclass),["%.3f"%i for i in perf1])
    plt.xticks(range(nclass),["%.3f"%i for i in perf2])
    ncorrect=np.sum(errormap[range(nclass),range(nclass)])
    accu=float(ncorrect)/float(len(T))
    plt.title("total accuracy %.3f"%accu)

def Regression(X_train,X_test,Y_train,Y_test,regModel,multiband=0,doplot=0):
    '''
    :param Y:
    :param X:
    :param prec:
    :param regModel:
    :param multiband: 0: each band need to be train seperately. 1: train all bands at same time (neural network)
    :param doplot: if do plot
    :return:
    '''
    nband=Y_train.shape[1]
    Yp=np.zeros(Y_test.shape)
    Yptrain=np.zeros(Y_train.shape)
    regModelList=[None]*nband
    if multiband==0:
        for i in range(0, nband, 1):
            print("regressing band %i"%i)
            model=copy.copy(regModel)
            model.fit(X_train, Y_train[:,i])
            Yp[:,i] = model.predict(X_test)
            Yptrain[:,i] = model.predict(X_train)
            regModelList[i]=model
    elif multiband==1:
        regModel.fit(X_train, Y_train)
        Yp = regModel.predict(X_test)
        Yptrain = regModel.predict(X_train)
        regModelList=[regModel]
    Regression_plot(Yp,Y_test,doplot)
    return (Yp,Yptrain,regModelList)

def Regression_plot(Yp,Ytest,doplot):
    rmse,rmse_band=RMSECal(Yp,Ytest)
    if doplot==1:   # box plot for all bands
        nband = Yp.shape[1]
        plt.figure()
        plt.boxplot(Yp-Ytest)
        #plt.title("Pred - Truth, total rmse %.3f"%np.mean(rmse_band))
    elif doplot==2:
        nband = Yp.shape[1]
        n=nband
        f, axarr = plt.subplots(int(np.ceil(n/2)), 2)
        f.tight_layout()
        f.subplots_adjust(top=0.9)
        #plt.suptitle('Silhouette Coefficient=%s'%score)
        for i in range(n):
            pj=int(np.ceil(i/2))
            pi=int(i%2)
            axarr[pj, pi].plot(Ytest[:,i],Yp[:,i],'.')
            axarr[pj, pi].set_title('cluster %s,\n corrcoef=%.3f, rmse=%.3f'\
                                    %(i,np.corrcoef(Ytest[:,i],Yp[:,i])[0,1],rmse_band[i]))
            plotFun.plot121line(axarr[pj, pi])
        f.tight_layout()

def FeatureSelectForward(X_train,X_test,Y_train,Y_test,model,multiband=1,nsel=None,verbose=1):
    '''
    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param model:
    :param multiband: int. multiband in SSRS.Regression
    :param nsel: int. Default None: do ranking for all attributes. Else: find best nsel attributes.
    :return attr_sel: list. Ind of attributes from best to worst.
    :return score: float. rmse of each step
    :return scoreRef: float. rmse of each step / rmse if use all attributes.
    '''
    nattr=X_train.shape[1]
    if nsel is None:
        nsel=nattr
    attr_sel=[]
    attr_rem=range(0,nattr)
    score=[]
    scoreRef=[]
    n=0
    for j in range(0,nsel):
        n=n+1
        if verbose==1: print("step %s"%n)
        scoretemp=[-9999]*nattr
        for k in attr_rem:
            attr=attr_sel[:]
            attr.append(k)
            Yp,Yptrain,regModelList=Regression(X_train[:,attr],X_test[:,attr],Y_train,Y_test,
                                               model,multiband=multiband,doplot=0)
            rmse,rmse_band=RMSECal(Yp,Y_test)
            scoretemp[k]=rmse
        ind=scoretemp.index(max(scoretemp))
        attr_sel.append(ind)
        attr_rem.remove(ind)
        score.append(scoretemp[ind])
        Yp,Yptrain,regModelList=Regression(X_train[:,attr_sel],X_test[:,attr_sel],Y_train,Y_test,
                                           model,multiband=multiband,doplot=0)
        rmse,rmse_band=RMSECal(Yptrain,Y_train)
        scoreRef.append(rmse)
    return attr_sel,score,scoreRef

def FeatureSelectBackward(X_train,X_test,Y_train,Y_test,model,multiband=1):
    '''
    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param model:
    :param multiband: int. multiband in SSRS.Regression
    :return attr_rem: list. Ind of attributes from worst to best.
    :return score: float. rmse of each step
    :return scoreRef: float. rmse of each step / rmse if use all attributes.
    '''
    nattr=X_train.shape[1]
    attr_sel=range(0,nattr)
    attr_rem=[]
    score=[]
    scoreRef=[]
    n=1 # first step: regression from all attributes
    print("step %s"%n)
    model.fit(X_train, Y_train)
    score.append(model.score(X_test,Y_test))
    Yp,Yptrain,regModelList=Regression(X_train[:,attr_sel],X_test[:,attr_sel],Y_train,Y_test,
                                       model,multiband=multiband,doplot=0)
    rmse,rmse_band=RMSECal(Yptrain,Y_train)
    scoreRef.append(rmse)
    for j in range(0,nattr-1):
        n=n+1
        print("step %s"%n)
        scoretemp=[-9999]*nattr
        for k in attr_sel:
            attr=attr_sel[:]
            attr.remove(k)
            Yp,Yptrain,regModelList=\
                Regression(X_train[:,attr],X_test[:,attr],Y_train,Y_test,model,multiband=multiband,doplot=0)
            rmse,rmse_band=RMSECal(Yp,Y_test)
            scoretemp[k]=rmse
        ind=scoretemp.index(max(scoretemp))
        attr_sel.remove(ind)
        attr_rem.append(ind)
        score.append(scoretemp[ind])
    Yp,Yptrain,regModelList=Regression(X_train[:,attr_sel],X_test[:,attr_sel],Y_train,Y_test,
                                       model,multiband=multiband,doplot=0)
    rmse,rmse_band=RMSECal(Yptrain,Y_train)
    scoreRef.append(rmse)
    attr_rem.append(attr_sel[0])    # append the last one
    return attr_rem,score,scoreRef

def FeatureSelect_plot(X_train,X_test,Y_train,Y_test,model,opt=0,figname=[]):
    #opt=0:forward; opt=1:backward
    nattr=X_train.shape[1]
    nband=Y_train.shape[1]
    if nband>10:
        nf=[10]*int(math.ceil(nband/10))
        nf.append(nband%10)
    else:
        nf=[nband]
    k=0
    indf=0
    attr_all=np.ndarray([nattr,nband])
    score_all=np.ndarray([nattr,nband])
    scoreRef_all=np.ndarray([nattr,nband])
    for n in nf:
        indf=indf+1
        f, axarr = plt.subplots(int(np.ceil(n/2)), 2)
        f.tight_layout()
        f.subplots_adjust(top=0.9)
        #plt.suptitle('Silhouette Coefficient=%s'%score)
        for i in range(n):
            print("calculating band%s"%k)
            if opt==0:
                attr,score,scoreRef=FeatureSelectForward\
                    (X_train,Y_train[:,k],X_test,Y_test[:,k],model)
            elif opt==1:
                attr,score,scoreRef=FeatureSelectBackward\
                    (X_train,Y_train[:,k],X_test,Y_test[:,k],model)
            pj=int(np.ceil(i/2))
            pi=int(i%2)
            axarr[pj, pi].plot(score)
            axarr[pj, pi].plot(scoreRef)
            axarr[pj, pi].set_title('object %s'%k)
            attr_all[:,k]=attr
            score_all[:,k]=score
            scoreRef_all[:,k]=scoreRef
            k=k+1
        if len(figname)!=0:
            if len(nf)==1:
                plt.savefig(figname)
            else:
                plt.savefig(figname+"f%s"%indf)
    return attr_all,score_all,scoreRef_all

def DistPlot(Xn,dist,figname,field):
    '''
    plot and save predictor vs distance figures
    :param Xn: numpy.ndarray. predictors, nind * nattr
    :param dist: numpy.ndarray. distance (or other thing), nind * ncenter(nabnd)
    :param figname: string. figname_#attr.png
    :param field: string. field name and show as title
    :return:
    '''
    nind,nattr=Xn.shape
    nind,nc=dist.shape
    for j in range(nattr):
        plt.close('all')
        f, axarr = plt.subplots(int(np.ceil(nc / 2)), 2)
        f.tight_layout()
        f.subplots_adjust(top=0.9)
        ff=field[j].tolist()
        ff=ff[0]
        plt.suptitle('%s'%ff)
        for i in range(nc):
            pj = int(np.ceil(i / 2))
            pi = int(i % 2)
            axarr[pj, pi].plot(Xn[:,j],dist[:,i],'.')
            cf=np.corrcoef(Xn[:,j],dist[:,i])
            axarr[pj, pi].set_title('cluster %s, coef=%.3f' % (i + 1, cf[0,1]))
        figfile=figname+'%s'%j
        plt.savefig(figfile)

def TraverseTree(regTree,Xin,Fields=None):
    '''
    :param regTree: tree.DecisionTreeRegressor.tree_
    :param Fields: list. Of input feature name.
    :param Xin:numpy.ndarray. Attribute data, of size nind * nattr
    :return string: list of string: String[i]: division rule of node i
    :return nodeind: list of list: nodeind[i]: individual indexes of node i
    :return leaf: list: leaf node indexes.
    :return label: numpy.ndarray: 1D array of size of nind. Label each individual by leaf node index.
    '''
    nnode = regTree.node_count
    if Fields is not None:
        featurename=[Fields[i] for i in regTree.feature]
    else:
        featurename=["Fields%i"%i for i in regTree.feature]
    string=[""]*nnode
    nodeind=[None]*nnode
    nind=Xin.shape[0]
    leaf=[]
    label=np.zeros([nind])
    def recurse(tempstr,node,Xtemp,indtemp):
        string[node]="node#%i: "%node+tempstr
        nodeind[node]=indtemp
        if (regTree.threshold[node] != -2):
            if regTree.children_left[node] != -1:
                tempstr= tempstr + " ( " + featurename[node] + " <= " + "%.3f"%(regTree.threshold[node]) + " ) ->\n "
                indlocal=np.where(Xtemp[:,regTree.feature[node]]<=regTree.threshold[node])[0]
                indleft=[indtemp[i] for i in indlocal]
                Xleft=Xtemp[indlocal,:]
                recurse(tempstr,regTree.children_left[node],Xleft,indleft)
            if regTree.children_right[node] != -1:
                tempstr= " ( " + featurename[node] + " > " + "%.3f"%(regTree.threshold[node]) + " ) ->\n "
                indlocal=np.where(Xtemp[:,regTree.feature[node]]>regTree.threshold[node])[0]
                indright=[indtemp[i] for i in indlocal]
                Xright=Xtemp[indlocal,:]
                recurse(tempstr,regTree.children_right[node],Xright,indright)
        else:
            leaf.append(node)
            label[indtemp]=node

    recurse("",0,Xin,range(0,nind))
    return string,nodeind,leaf,label

def TraverseTreePlot(regTree,Xin,Fields=None):
    '''
    :param regTree: tree.DecisionTreeRegressor.tree_
    :param Fields: list. Of input feature name.
    :param Xin:numpy.ndarray. Attribute data, of size nind * nattr
    :return string: list of string: String[i]: division rule of node i
    :return nodeind: list of list: nodeind[i]: individual indexes of node i
    :return leaf: list: leaf node indexes.
    :return label: numpy.ndarray: 1D array of size of nind. Label each individual by leaf node index.
    '''
    nnode = regTree.node_count
    if Fields is not None:
        featurename=[Fields[i] for i in regTree.feature]
    else:
        featurename=["Fields%i"%i for i in regTree.feature]
    string=[""]*nnode
    nodeind=[None]*nnode
    nind=Xin.shape[0]
    leaf=[]
    label=np.zeros([nind])
    aboveNodeList=[None]*nnode
    aboveNodeDirList=[None]*nnode
    def recurse(tempstr,node,Xtemp,indtemp,abovenode):
        print(node)
        print(abovenode)
        aboveNodeList[node]=abovenode
        abovenode=np.append(abovenode,node)
        string[node]="node#%i: "%node+tempstr
        nodeind[node]=indtemp
        if (regTree.threshold[node] != -2):
            if regTree.children_left[node] != -1:
                tempstr= tempstr + " ( " + featurename[node] + " <= " + "%.3f"%(regTree.threshold[node]) + " ) ->\n "
                indlocal=np.where(Xtemp[:,regTree.feature[node]]<=regTree.threshold[node])[0]
                indleft=[indtemp[i] for i in indlocal]
                Xleft=Xtemp[indlocal,:]
                recurse(tempstr,regTree.children_left[node],Xleft,indleft,abovenode)
            if regTree.children_right[node] != -1:
                tempstr= " ( " + featurename[node] + " > " + "%.3f"%(regTree.threshold[node]) + " ) ->\n "
                indlocal=np.where(Xtemp[:,regTree.feature[node]]>regTree.threshold[node])[0]
                indright=[indtemp[i] for i in indlocal]
                Xright=Xtemp[indlocal,:]
                recurse(tempstr,regTree.children_right[node],Xright,indright,abovenode)
        else:
            leaf.append(node)
            label[indtemp]=node

    recurse("",0,Xin,range(0,nind))

    # plot from bottom
    gs = gridspec.GridSpec(regTree.max_depth*3+4,leaf.__len__()*4)
    depth=np.zeros(nnode)
    isleaf=np.zeros(nnode)
    for i in range(0,nnode):
        depth[i]=aboveNodeList[i].shape[0]
        if i in leaf:
            isleaf[i]=1

    plt.subplot(gs[3:5,3:5])

    return string,nodeind,leaf,label

def RegressionTree(X_train,X_test,Y_train,Y_test,regTreeModel,fitModel,Field,
                   doFitSelection=0,doMultiBand=0):
    '''
    1. do regression tree
    2. inside each leaf, do linear regression from input attributes(X)
    :param X_train: numpy.ndarray
    :param X_test: numpy.ndarray
    :param Y_train: numpy.ndarray
    :param Y_test: numpy.ndarray
    :param regTree: tree.DecisionTreeRegressor
    :param doFitSelection: int. if use tree predictors in linear regression, =0.
                                if do feature selection in linear regression, = # of selected feature.
    :return Yp: numpy.ndarray
    :return Yptrain: numpy.ndarray
    :return regTreeModel: tree.DecisionTreeRegressor
    :return fitModelList: list. fit model list, size of #leaves from tree
    :return predind: numpy.ndarray. index of involved predictors1d. 1d array.
    '''
    ntest,nband=Y_test.shape

    Yptree,Yptraintree,regTreeModelList=Regression\
        (X_train,X_test,Y_train,Y_test,multiband=doMultiBand,regModel=regTreeModel,doplot=0)

    if doMultiBand==0:
        nit=nband
        predindList=[None]*nband
        fitModelList2d=[None]*nband
    elif doMultiBand==1:
        nit=1

    for ii in range(0,nit):
        regTree=regTreeModelList[ii].tree_
        # retrain leaf by trainning dataset
        string,nodeind,leaf,label=TraverseTree(regTree,X_train,Field)
        if doFitSelection==0:
            predind=np.delete(np.unique(regTree.feature),0)
        else:
            predind=np.zeros([leaf.__len__(),doFitSelection])
        fitModelList=[None]*leaf.__len__()
        k=0
        Yptrain=Y_train*0
        for i in leaf:
            model=copy.copy(fitModel)
            Xtemp=X_train[label==i]
            Ytemp=Y_train[label==i]
            if doFitSelection==0:
                Xtemp=Xtemp[:,predind]
            else:
                attr_sel,score,scoreRef=FeatureSelectForward(Xtemp,Ytemp,Xtemp,Ytemp,model,multiband=1,nsel=doFitSelection,verbose=0)
                predind[k,:]=attr_sel[0:doFitSelection]
                Xtemp=Xtemp[:,predind[k,:].astype(int)]
            if Xtemp.shape[0]!=0 and Xtemp.shape[1]!=0:
                model.fit(Xtemp, Ytemp)
                Yptemp=model.predict(Xtemp)
                Yptrain[label==i]=Yptemp
                fitModelList[k]=model
            k=k+1
        #apply to test dataset
        string,nodeind,leaftest,labeltest=TraverseTree(regTree,X_test,Field)
        Yp=Y_test*0
        k=0
        for i in leaftest:
            Xtemp=X_test[labeltest==i]
            if doFitSelection==0:
                Xtemp=Xtemp[:,predind]
            else:
                Xtemp=Xtemp[:,predind[k,:].astype(int)]
            model=fitModelList[k]
            if Xtemp.shape[0]!=0 and Xtemp.shape[1]!=0:
                Yptemp=model.predict(Xtemp)
                Yp[labeltest==i]=Yptemp
            k=k+1
        if doMultiBand==0:
            predindList[ii]=predind
            fitModelList2d[ii]=fitModelList
    if doMultiBand==0:
        return(Yp,Yptrain,regTreeModelList,fitModelList2d,predindList)
    elif doMultiBand==1:
        regTreeModel=regTreeModelList[0]
        return(Yp,Yptrain,regTreeModel,fitModelList,predind)

def FeatureSelectForward_RegTree(X_train,X_test,Y_train,Y_test,regTreeModel,fitModel,Field,\
                                 doFitSelection=0,doMultiBand=0,nsel=None,verbose=1):
    '''
    Select features for
    :param X_train: numpy.ndarray
    :param X_test: numpy.ndarray
    :param Y_train: numpy.ndarray
    :param Y_test: numpy.ndarray
    :param regTree: tree.DecisionTreeRegressor
    :param doFitSelection: int. if use tree predictors in linear regression, =0.
                                if do feature selection in linear regression, = # of selected feature.
    :param multiband: int. multiband in SSRS.Regression
    :param nsel: int. Default None: do ranking for all attributes. Else: find best nsel attributes.
    :return attr_sel: list. Ind of attributes from best to worst.
    :return score: float. rmse of each step
    :return scoreRef: float. rmse of each step / rmse if use all attributes.
    '''
    nattr=X_train.shape[1]
    if nsel is None:
        nsel=nattr
    attr_sel=[]
    attr_rem=range(0,nattr)
    score=[]
    scoreRef=[]
    n=0
    for j in range(0,nsel):
        n=n+1
        if verbose==1: print("step %s"%n)
        scoretemp=[9999]*nattr
        for k in attr_rem:
            attr=attr_sel[:]
            attr.append(k)
            Yp,Yptrain,regTreeModel,fitModelList,predind=\
                RegressionTree(X_train[:,attr],X_test[:,attr],Y_train,Y_test,regTreeModel,fitModel,Field,
                               doFitSelection=doFitSelection,doMultiBand=doMultiBand)
            rmse,rmse_band=RMSECal(Yp,Y_test)
            scoretemp[k]=rmse
        ind=scoretemp.index(min(scoretemp))
        print(ind)
        attr_sel.append(ind)
        attr_rem.remove(ind)
        print(attr_rem)
        score.append(scoretemp[ind])
        Yp,Yptrain,regTreeModel,fitModelList,predind=\
            RegressionTree(X_train[:,attr_sel],X_test[:,attr_sel],Y_train,Y_test,regTreeModel,fitModel,Field,
                           doFitSelection=doFitSelection,doMultiBand=doMultiBand)
        rmse,rmse_band=RMSECal(Yptrain,Y_train)
        scoreRef.append(rmse)
    return attr_sel,score,scoreRef

def FeatureSelectBackward_RegTree(X_train,X_test,Y_train,Y_test,regTreeModel,fitModel,Field,\
                                  doFitSelection=0,doMultiBand=0,):
    '''
    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param model:
    :param multiband: int. multiband in SSRS.Regression
    :return attr_rem: list. Ind of attributes from worst to best.
    :return score: float. rmse of each step
    :return scoreRef: float. rmse of each step / rmse if use all attributes.
    '''
    nattr=X_train.shape[1]
    attr_sel=range(0,nattr)
    attr_rem=[]
    score=[]
    scoreRef=[]
    n=1 # first step: regression from all attributes
    print("step %s"%n)
    Yp,Yptrain,regTreeModel,fitModelList,predind=\
        RegressionTree(X_train,X_test,Y_train,Y_test,regTreeModel,fitModel,Field,
                       doFitSelection=doFitSelection,doMultiBand=doMultiBand)
    rmse,rmse_band=RMSECal(Yp,Y_test)
    score.append(rmse)
    rmse,rmse_band=RMSECal(Yptrain,Y_train)
    scoreRef.append(rmse)
    for j in range(0,nattr-1):
        n=n+1
        print("step %s"%n)
        scoretemp=[9999]*nattr
        for k in attr_sel:
            attr=attr_sel[:]
            attr.remove(k)
            Yp,Yptrain,regTreeModel,fitModelList,predind=\
                RegressionTree(X_train[:,attr],X_test[:,attr],Y_train,Y_test,regTreeModel,fitModel,Field,
                               doFitSelection=doFitSelection,doMultiBand=doMultiBand)
            rmse,rmse_band=RMSECal(Yp,Y_test)
            scoretemp[k]=rmse
        ind=scoretemp.index(min(scoretemp))
        print(ind)
        print(attr_sel)
        attr_sel.remove(ind)
        attr_rem.append(ind)
        score.append(scoretemp[ind])
        Yp,Yptrain,regTreeModel,fitModelList,predind=\
            RegressionTree(X_train[:,attr_sel],X_test[:,attr_sel],Y_train,Y_test,regTreeModel,fitModel,Field,
                           doFitSelection=doFitSelection,doMultiBand=doMultiBand)
        rmse,rmse_band=RMSECal(Yptrain,Y_train)
        scoreRef.append(rmse)
    attr_rem.append(attr_sel[0])    # append the last one
    return attr_rem,score,scoreRef

def TreePlot(savedir,X_train,X_test,Y_train,Y_test,regTreeModel,Field,fitModelList=None,predind=None):
    ## plot tree with a regression model

    regTree=regTreeModel.tree_

    # plot tree. need pydot, pyparser, graphviz... environment setted up in my Mac.
    from sklearn import tree
    with open(savedir+"tree.dot", 'w') as f:
        f = tree.export_graphviz(regTreeModel, out_file=f,feature_names=Field,
                                 label='none',node_ids=True)
    # os.system("dot -Tpng tree.dot -o tree.png")

    #plot each node for test
    def traverseplot(Xin,Yin,Field,name):
        string,nodeind,leaf,label=TraverseTree(regTree,Xin,Field)
        nband=Yin.shape[1]
        k=0
        for j in leaf:
            Ytemp=Yin[nodeind[j],:]
            Xtemp=Xin[nodeind[j],:]
            Yptemp=regTreeModel.predict(Xtemp)

            fitmodel=fitModelList[k]
            if predind.ndim==1:
                Ypnewtemp=fitmodel.predict(Xtemp[:,predind.astype(int)])
            else:
                Ypnewtemp=fitmodel.predict(Xtemp[:,predind[k,:].astype(int)])

            rmse,rmse_band=RMSECal(Yptemp,Ytemp)
            rmsenew,rmse_bandnew=RMSECal(Ypnewtemp,Ytemp)

            n=nband
            f, axarr = plt.subplots(int(np.ceil(n/2)), 2,figsize=(10,12))
            for i in range(n):
                pj=int(np.ceil(i/2))
                pi=int(i%2)
                axarr[pj, pi].plot(Yptemp[:,i],Ytemp[:,i],'.')
                axarr[pj, pi].plot(Ypnewtemp[:,i],Ytemp[:,i],'.r')
                axarr[pj, pi].set_title('cluster %s,\n cc=%.3f -> %.3f, r=%.3f -> %.3f'\
                                        %(i,np.corrcoef(Yptemp[:,i],Ytemp[:,i])[0,1],np.corrcoef(Ypnewtemp[:,i],Ytemp[:,i])[0,1],
                                          rmse_band[i],rmse_bandnew[i]))
                plotFun.plot121line(axarr[pj, pi])
            f.tight_layout()
            f.suptitle(string[j],fontsize=8)
            f.subplots_adjust(top=0.9)
            plt.savefig(savedir+name+"_node%i"%j)
            plt.close()
            k=k+1

    traverseplot(X_train,Y_train,Field,"Train")
    traverseplot(X_test,Y_test,Field,"Test")

def Tree2Mat(regTree,predSel,Field,treeMatFile,X,Xn,ind_train,ind_test,Y_train,Y_test,Yp,Yptrain):
    predName=[Field[i] for i in predSel]

    # calculate actual threshold.
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

    string_all,nodeind_all,leaf_all,label_all=\
        TraverseTree(regTree, Xn[:,predSel], predName)
    string_train,nodeind_train,leaf_train,label_train=\
        TraverseTree(regTree,Xn[ind_train,:][:,predSel],predName)
    string_test,nodeind_test,leaf_test,label_test=\
        TraverseTree(regTree,Xn[ind_test,:][:,predSel],predName)

    sio.savemat(treeMatFile,
                {'predSel':predSel,
                 "nodeind":nodeind_all,"nodeind_train":nodeind_train,
                 "nodeind_test":nodeind_test,"the":the,
                 "ind_train":ind_train,"ind_test":ind_test,
                 "cleft":regTree.children_left,"cright":regTree.children_right,
                 "fieldInd":fieldInd,"nodeValue":nodeValue,
                 "Yp":Yp,"Yptrain":Yptrain,
                 "Y_train":Y_train,"Y_test":Y_test})


def RMSECal(x,y):
    if x.shape.__len__()==1:
        x = np.reshape(x, (-1, 1))

    nband=x.shape[1]
    rmse_band=np.zeros([nband,1])
    for i in range(0, nband, 1):
        rmse_band[i] = np.sqrt(np.mean((x[:,i] - y[:,i])**2, axis=0))
    rmse=rmse_band.mean()
    return(rmse,rmse_band)

def DistCal(a,b):
    nv=np.where(np.isfinite(a-b))[0].shape[0]
    n=a.shape[0]
    d=np.square(a-b)
    dist=np.sqrt(np.nansum(d))/nv*n
    return(dist)

