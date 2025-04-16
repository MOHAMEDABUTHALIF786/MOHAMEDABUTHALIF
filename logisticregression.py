
"""----------------------------------------------------------------------------"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris,load_wine
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error, root_mean_squared_error, mean_absolute_error
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

model=LogisticRegression()








