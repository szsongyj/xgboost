#coding:utf-8

#-------------------------------------------------------------------------------
# Description:  xgboost with python
# Project Name: xgboost
# Name:         early-stopping-xgboost
# Author:       szsongyj
# DateTime:     2019/7/8 21:25
#-------------------------------------------------------------------------------

# early stopping
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)
# fit model on training data
model = XGBClassifier()
eval_set = [(X_test, y_test)]
# specify a window of the number of epochs over which no improvement is observed.
# This is specified in the early stopping rounds parameter.
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss",
eval_set=eval_set, verbose=True)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
