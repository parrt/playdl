import numpy as np
import pandas as pd

import statsmodels.api as sm
from scipy.stats import spearmanr

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils import resample
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype, \
    is_bool_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, \
    confusion_matrix
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, f1_score, accuracy_score,\
                            roc_auc_score, average_precision_score, precision_recall_curve, auc,\
                            roc_curve
import xgboost as xgb

import tensorflow as tf
import keras
from keras.datasets import mnist

from timeit import default_timer as timer
from collections import OrderedDict
import os

(X_train, y_train), (X_test, y_test) = mnist.load_data()

sub = 10_000
X_train = X_train[:sub,:,:]
X_test = X_test[:sub,:,:]
y_train = y_train[:sub]
y_test = y_test[:sub]

n, w, h = X_train.shape

print(n, w, h)

# plt.imshow(X_train[0], cmap='gray')
# plt.show()

rf = RandomForestClassifier(n_estimators=200,
                            min_samples_leaf=2,
                            oob_score=True, n_jobs=-1)
rf.fit(X_train.reshape(n,-1), y_train)
print(rf.oob_score_)

y_pred = rf.predict(X_test.reshape(n, -1))
conf = confusion_matrix(y_test, y_pred)
print(conf)
print("test accuracy", accuracy_score(y_test, y_pred))
# f1 = f1_score()
# print("f1", f1)