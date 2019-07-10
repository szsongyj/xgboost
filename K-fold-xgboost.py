#coding:utf-8

#-------------------------------------------------------------------------------
# Description:  Machine Learning Algorithm Practice
# Project Name: PycharmProjects
# Name:         K-fold-xgboost
# Author:       0049003103
# DateTime:     2019/7/8 16:21
#-------------------------------------------------------------------------------
# k-fold cross validation evaluation of xgboost model
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# CV model
model = XGBClassifier()
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))