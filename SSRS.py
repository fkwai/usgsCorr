
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn import metrics
from sklearn import cross_validation

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
    f, axarr = plt.subplots(int(np.ceil(n/2)), 2)
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

def Regression(Xtrain,Xtest,Ytrain,Ytest,regModel,multiband=0,doplot=0):
    '''
    :param Y:
    :param X:
    :param prec:
    :param regModel:
    :param multiband: 0: each band need to be train seperately. 1: train all bands at same time (neural network)
    :param doplot: if do plot
    :return:
    '''
    nband=Ytrain.shape[1]

    rmse_band=np.zeros([nband,1])
    rmse_band_train=np.zeros([nband,1])
    Yp=np.zeros(Ytest.shape)
    Yptrain=np.zeros(Ytrain.shape)

    if multiband==0:
        for i in range(0, nband, 1):
            print("regressing band %i"%i)
            regModel.fit(Xtrain, Ytrain[:,i])
            Yp[:,i] = regModel.predict(Xtest)
            Yptrain[:,i] = regModel.predict(Xtrain)
            rmse_band[i] = np.sqrt(np.mean((Yp[:,i] - Ytest[:,i])**2, axis=0))
            rmse_band_train[i] = np.sqrt(np.mean((Yptrain[:,i] - Ytrain[:,i])**2, axis=0))
    elif multiband==1:
        regModel.fit(Xtrain, Ytrain)
        Yp = regModel.predict(Xtest)
        Yptrain = regModel.predict(Xtrain)
        for i in range(0, nband, 1):
            rmse_band[i] = np.sqrt(np.mean((Yp[:,i] - Ytest[:,i])**2, axis=0))
            rmse_band_train[i] = np.sqrt(np.mean((Yptrain[:,i] - Ytrain[:,i])**2, axis=0))

    Regression_plot(Yp,Ytest,doplot)
    rmse=rmse_band.mean()
    rmse_train=rmse_band_train.mean()
    return (Yp,rmse,rmse_train,rmse_band,rmse_band_train)

def Regression_plot(Yp,Ytest,doplot):
    nband=Yp.shape[1]
    if doplot==1:   # box plot for all bands
        plt.boxplot(Yp-Ytest)
        #plt.title("Pred - Truth, total rmse %.3f"%np.mean(rmse_band))
    elif doplot==2:
        n=nband
        f, axarr = plt.subplots(int(np.ceil(n/2)), 2)
        f.tight_layout()
        f.subplots_adjust(top=0.9)
        #plt.suptitle('Silhouette Coefficient=%s'%score)
        for i in range(n):
            pj=int(np.ceil(i/2))
            pi=int(i%2)
            axarr[pj, pi].plot(Ytest[:,i],Yp[:,i],'.')
            axarr[pj, pi].set_title('cluster %s, corrcoef=%.3f'\
                                    %(i,np.corrcoef(Ytest[:,i],Yp[:,i])))
            plotFun.plot121line(axarr[pj, pi])

def FeatureSelectForward(X_train,Y_train,X_test,Y_test,model,multiband=1):
    X_all=np.append(X_train,X_test,axis=0)
    Y_all=np.append(Y_train,Y_test,axis=0)
    nattr=X_train.shape[1]
    attr_sel=[]
    attr_rem=range(0,nattr)
    score=[]
    scoreRef=[]
    n=0
    for j in range(0,nattr):
        n=n+1
        print("step %s"%n)
        scoretemp=[-9999]*nattr
        for k in attr_rem:
            attr=attr_sel[:]
            attr.append(k)
            Yp,rmse,rmse_train,rmse_band,rmse_band_train=\
                Regression(X_train[:,attr],X_test[:,attr],Y_train,Y_test,model,multiband=multiband,doplot=0)
            scoretemp[k]=rmse
        ind=scoretemp.index(max(scoretemp))
        attr_sel.append(ind)
        attr_rem.remove(ind)
        score.append(scoretemp[ind])
        model.fit(X_all[:,attr_sel], Y_all)
        scoreRef.append(model.score(X_all[:,attr_sel],Y_all))
    return attr_sel,score,scoreRef

def FeatureSelectBackward(X_train,Y_train,X_test,Y_test,model,multiband=1):
    X_all=np.append(X_train,X_test,axis=0)
    Y_all=np.append(Y_train,Y_test,axis=0)
    nattr=X_train.shape[1]
    attr_sel=range(0,nattr)
    attr_rem=[]
    score=[]
    scoreRef=[]
    n=1 # first step: regression from all attributes
    print("step %s"%n)
    model.fit(X_train, Y_train)
    score.append(model.score(X_test,Y_test))
    model.fit(X_all, Y_all)
    scoreRef.append(model.score(X_all,Y_all))
    for j in range(0,nattr-1):
        n=n+1
        print("step %s"%n)
        scoretemp=[-9999]*nattr
        for k in attr_sel:
            attr=attr_sel[:]
            attr.remove(k)
            Yp,rmse,rmse_train,rmse_band,rmse_band_train=\
                Regression(X_train[:,attr],X_test[:,attr],Y_train,Y_test,model,multiband=multiband,doplot=0)
            scoretemp[k]=rmse
        ind=scoretemp.index(max(scoretemp))
        attr_sel.remove(ind)
        attr_rem.append(ind)
        score.append(scoretemp[ind])
        model.fit(X_all[:,attr_sel], Y_all)
        scoreRef.append(model.score(X_all[:,attr_sel],Y_all))
    attr_rem.append(attr_sel[0])    # append the last one
    return attr_rem,score,scoreRef

def FeatureSelect_plot(X_train,Y_train,X_test,Y_test,model,opt=0,figname=[]):
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
    # plot and save predictor vs distance figures
    # Xn: predictors, nind * nattr
    # dist: distance (or other thing), nind * ncenter
    # field: field name and show as title
    # figure save as figname_#attr.png
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

def traverseTree(regTree, feature_names,Xin):
    featurename=[feature_names[i] for i in regTree.feature]
    nnode = regTree.node_count
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
                recurse (tempstr,regTree.children_left[node],Xleft,indleft)
            if regTree.children_right[node] != -1:
                tempstr= " ( " + featurename[node] + " > " + "%.3f"%(regTree.threshold[node]) + " ) ->\n "
                indlocal=np.where(Xtemp[:,regTree.feature[node]]>regTree.threshold[node])[0]
                indright=[indtemp[i] for i in indlocal]
                Xright=Xtemp[indlocal,:]
                recurse (tempstr,regTree.children_right[node],Xright,indright)
        else:
            leaf.append(node)
            label[indtemp]=node

    recurse("",0,Xin,range(0,nind))
    return string,nodeind,leaf,label