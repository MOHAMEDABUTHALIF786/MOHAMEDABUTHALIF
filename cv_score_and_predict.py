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
from sklearn.metrics import accuracy_score,r2_score,rand_score
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
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
cv_predict=cross_val_predict


y_predicted=cv_predict(tree,x,y,cv=3)

print("y_predicted:",y_predicted)

accuracy_score =cross_val_score(tree,x,y,cv=3,scoring="accuracy",)
print("cross_val_score:",accuracy_score)

avg_accuracy_score=accuracy_score.mean()
print("cross_val_score.avg:",avg_accuracy_score)

# accuracy_check1 =r2_score(y_test,y_predicted)
# print("r2score:",accuracy_check1)

# accuracy_check2=accuracy_score(y_test,y_predicted)
# print("accuracy_score:",accuracy_check2)

# accuracy_check3 =mean_squared_error(y_test,y_prediction)
# print("mean_squared_error:",accuracy_check3)
#
# accuracy_check4=mean_absolute_error(y_test,y_prediction)
# print("mean_absolute_error",accuracy_check4)

"""
y_predicted was decision tree regression():

y_predicted: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1
 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 2 2 2 2
 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
cross_val_score: [0.98 0.92 0.96]
cross_val_score.avg: 0.9533333333333333
_________________________________________________________________________________________________________________
 y_predicded was decision tree classifier():
 
 y_predicted: 
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1
 1 1 1 2 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 2 2 2 2
 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
cross_val_score: [0.98 0.94 0.98]
cross_val_score.avg: 0.9666666666666667

 
"""




