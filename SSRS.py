
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import metrics


def Cluster(data,model,doplot=1):
    model.fit(data)
    label=model.labels_
    if hasattr(model, 'cluster_centers_'):
        center=model.cluster_centers_
    else:
        center=[]
    if doplot==1:
        cluster_plot(data,label,center)
    return label,center

def cluster_plot(data,label,center,rank=0):
    nclass=len(np.unique(label))
    nind_class=np.bincount(label)
    if rank==1:
        sortind=np.argsort(nind_class)[::-1]
    else:
        sortind=range(0,nclass)
    score=metrics.silhouette_score(data,label)

    if nclass>10:
        n=10
    else:
        n=nclass
    f, axarr = plt.subplots(int(np.ceil(n/2)), 2)
    f.tight_layout()
    f.subplots_adjust(top=0.9)
    plt.suptitle('Silhouette Coefficient=%s'%score)
    for i in range(n):
        ind=sortind[i]
        pj=int(np.ceil(i/2))
        pi=int(i%2)
        axarr[pj, pi].boxplot(data[label==ind])
        axarr[pj, pi].set_title('cluster %s, nind=%s'%(ind,nind_class[ind]))
        axarr[pj, pi].set_ylim([-1,1])
        if len(center)>0:
            axarr[pj, pi].plot(range(1,31),center[ind,],'*-r')

def cluster_rename(label,ythe,Cpca,center=[]):
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

def DistPlot(Xn,dist,figname,field):
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