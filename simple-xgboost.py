#coding:utf-8
#-------------------------------------------------------------------------------
# Description:  Machine Learning Algorithm Practice
# Project Name: PycharmProjects
# Name:         simple-xgboost
# Author:       0049003103
# DateTime:     2019/7/5 14:49
#-------------------------------------------------------------------------------

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_tree
from xgboost import plot_importance
from matplotlib import pyplot
import pickle


dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,random_state=seed)

# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)

# visualization of model
plot_tree(model,num_trees=0,rankdir='LR')
pyplot.show()

# feature importance
print(model.feature_importances_)
# visualization of feature importance
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()

# plot of feature importance using built-in function of xgboost
plot_importance(model)
pyplot.show()

# save model to file
pickle.dump(model, open("pima.pickle.dat", "wb"))
print("Saved model to: pima.pickle.dat")

# load model from file
loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
print("Loaded model from: pima.pickle.dat")

# make predictions for test data
y_pred = model.predict(X_test)
y_pred_loadedmodel = loaded_model.predict(X_test)
predictions_loadedmodel = [round(value) for value in y_pred_loadedmodel]
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
accuracy_loadedmodel = accuracy_score(y_test,predictions_loadedmodel)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Accuracy from loadedmodel: %.2f%%" % (accuracy_loadedmodel * 100.0))
print(predictions_loadedmodel)
print(predictions)



