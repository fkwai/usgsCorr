
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import metrics


def Cluster(data,model):
    model.fit(data)
    label=model.labels_
    if hasattr(model, 'cluster_centers_'):
        center=model.cluster_centers_
    else:
        center=[]

    nclass=len(np.unique(label))
    nind_class=np.bincount(label)
    sortind=np.argsort(nind_class)[::-1]
    score=metrics.silhouette_score(data,label)

    if nclass>10:
        n=10
    else:
        n=nclass
    f, axarr = plt.subplots(int(np.ceil(n/2)), 2)
    plt.suptitle('Silhouette Coefficient=%s'%score)
    for i in range(n):
        ind=sortind[i]
        pj=int(np.ceil(i/2))
        pi=int(i%2)
        axarr[pj, pi].boxplot(data[label==ind])
        axarr[pj, pi].set_title('cluster %s, nind=%s'%(ind+1,nind_class[ind]))
        axarr[pj, pi].set_ylim([-1,1])
        if len(center)>0:
            axarr[pj, pi].plot(center[ind,],'*-r')
    return label

### training
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


def devideset(n,rate=0.2):
    indran=range(n)
    random.shuffle(indran)
    indran=np.asarray(indran)
    ind=int(np.round(n*rate))
    ind1=indran[range(0,ind,1)]
    ind2=indran[range(ind+1,n,1)]
    return(ind1,ind2)


def RegressionLearn(Y,X,prec,regModel):
    nind=Y.shape[0]
    nband=Y.shape[1]
    nattr=X.shape[1]

    [indtest,indtrain]=devideset(nind,prec)
    Ytest=Y[indtest,:]
    Xtest=X[indtest,:]
    Ytrain=Y[indtrain,:]
    Xtrain=X[indtrain,:]

    rmse_band=np.zeros([nband,1])
    Yp=np.zeros((len(indtest),nband))

    for i in range(0, nband, 1):
        print("regressing band %i"%i)
        regModel.fit(Xtrain, Ytrain[:,i])
        Yp[:,i] = regModel.predict(Xtest)
        rmse_band[i] = np.sqrt(np.mean((Yp[:,i] - Ytest[:,i])**2, axis=0))
    plt.boxplot(Yp-Ytest)
    plt.title("Pred - Truth, total rmse %.3f"%np.mean(rmse_band))
    return (rmse_band,Yp,Ytest)
