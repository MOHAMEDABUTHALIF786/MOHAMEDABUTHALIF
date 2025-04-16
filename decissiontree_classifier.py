
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris,load_wine
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import cross_val_score,cross_val_predict
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

tree=DecisionTreeClassifier()

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.4,random_state=42)

tree.fit(x_train,y_train)
y_prediction =tree.predict(x_test)
print("y_prediction:",y_prediction)

# accuracy_check1 =r2_score(y_test,y_prediction)
# print("r2score:",accuracy_check1)

# accuracy_check2=accuracy_score(y_test,y_prediction)
# print("accuracy_score:",accuracy_check2)

# accuracy_check3 =mean_squared_error(y_test,y_prediction)
# print("mean_squared_error:",accuracy_check3)
#
accuracy_check4=mean_absolute_error(y_test,y_prediction)
print("mean_absolute_error",accuracy_check4)

"""
y_prediction: 
[1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1
 0 0 0 2 1 1 0 0 1 1 2 1 2 1 2 1 0 2 1 0 0 0 1]
r2score: 0.9753593429158111
accuracy_score: 0.9666666666666667
mean_squared_error: 0.016666666666666666
mean_absolute_error 0.016666666666666666
"""

