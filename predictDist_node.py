import scipy.io as sio
from sklearn import linear_model
from sklearn import tree
import SSRS

# we want to split node inside given shape.

# 1.  load data
UCdir = "E:\Kuai\SSRS\paper\mB\nodeSplit\\"
Datafile=UCdir+"dataRegionB.mat"
mat = sio.loadmat(Datafile)
Field = [str(''.join(letter)) for letter_array in mat['field'] for letter in letter_array]
dataset=mat['dataset']
dist=mat['dist']




# predict dist
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