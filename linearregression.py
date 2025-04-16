

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error, root_mean_squared_error, mean_absolute_error,rand_score
from sklearn.cluster import k_means
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.manifold import TSNE
from scipy import stats
import scipy.stats as stats


data=load_iris()
x=data.data
y=data.target
model =LinearRegression()
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.4,random_state=42)

model.fit(x_train,y_train)

y_prediction =model.predict(x_test)
print("y_prediction:",y_prediction)

accuracy_check1 =r2_score(y_test,y_prediction)
print("r2score:",accuracy_check1)

# accuracy_check2=accuracy_score(y_test,y_prediction)
# print("accuracy_score:",accuracy_check2)

# accuracy_check3 =mean_squared_error(y_test,y_prediction)
# print("mean_squared_error:",accuracy_check3)
#
# accuracy_check4=mean_absolute_error(y_test,y_prediction)
# print("mean_absolute_error",accuracy_check4)


"""
y_prediction:
 [ 1.22763347 -0.04382918  2.23635896  1.35135194  1.2872452   0.01323033
  1.05428733  1.84332421  1.36152007  1.06170155  1.71506412 -0.09504909
 -0.16569824 -0.08533245 -0.02794852  1.40643346  2.01498994  1.03720055
  1.27729016  1.98440943  0.01389642  1.6057843   0.08332069  1.92833891
  1.86269578  1.8957576   1.79160099  2.05384509  0.01709193  0.0073867
 -0.14651477 -0.06658655  1.18247431 -0.00652585 -0.02916604  1.68824931
  1.29285043 -0.08125373 -0.08909816 -0.16514933  1.76021188  1.38423488
  1.31504901 -0.06096004 -0.11282324  0.94591201  1.4755561   1.72071343
  1.19825907  2.15384988  1.19614846  2.04031093  1.3913034  -0.09560788
  2.0078696   1.00056332 -0.10211761 -0.05389152  0.13012757  0.96919549]
  
r2score: 0.9421006573198892
mean_squared_error: 0.03916247206279712
mean_absolute_error 0.150642721934179

"""







