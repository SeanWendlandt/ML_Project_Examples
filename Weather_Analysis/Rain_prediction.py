# Sean Wendlandt Assignment 2


import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.naive_bayes import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.inspection import permutation_importance

# Data being used
df = pd.read_csv('assignment2_cleanInfile.csv')

print(df)
########### 1 ###################
# classifications are:
Y = df['RainTomorrow']
X = df.drop('RainTomorrow', axis=1)

print(Y)
print(X)

x_train, x_test, y_train, y_test  = train_test_split(X, Y,  test_size=0.25, random_state=2)
print('x_test = ' + str(x_test) )
print('y_test = ' + str(y_test) )



print("\n\nDecision Tree:")
dtree = tree.DecisionTreeClassifier(criterion="gini")
dtree = dtree.fit(x_train, y_train)
y_predicted = dtree.predict(x_test)
print('DecisionTree confusion matrix:')
print(confusion_matrix(y_test, y_predicted))
importance = dtree.feature_importances_
print("decision tree dtree feature importance:")
for i, v in enumerate(importance):
	print('Feature: %0d, FName: %15s, Score: %.5f' % (i, df.columns[i], v))
	# print('Feature: %0d, Score: %.5f' % (i,v))


print("\n\nGausianNB:")
model = GaussianNB()
model.fit(x_train, y_train)
gausianNB_predicted = model.predict(x_test)
print('\nconfusion_matrix from Gaussian naive bayes:')
print(confusion_matrix(y_test, gausianNB_predicted))
accuracy = accuracy_score(y_test, gausianNB_predicted)
print('accuracy = ' + str(accuracy))
imps = permutation_importance(model, x_test, y_test)
print("gaussinaNB feature importance:")
print(imps.importances_mean)

############### 2 #########################
# now drop RainToday attribute
print("\n\nDropping RainToday attribute")
X = X.drop('RainToday', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y,  test_size=0.25, random_state=2)
print(X)
print('x_test = ' + str(x_test))
print('y_test = ' + str(y_test))

print("\n\nDecision Tree:")
dtree2 = tree.DecisionTreeClassifier(criterion="gini")
dtree2 = dtree2.fit(x_train,y_train)
y_predicted = dtree2.predict(x_test)
print('DecisionTree confusion matrix:')
print(confusion_matrix( y_test, y_predicted))
importance = dtree2.feature_importances_
print("decision tree dtree2 feature importance:")
for i, v in enumerate(importance):
	print('Feature: %0d, FName: %15s, Score: %.5f' % (i, X.columns[i], v))

print("\n\nGausianNB:")
model2 = GaussianNB()
model2.fit(x_train,y_train)
gausianNB_predicted = model2.predict(x_test)
print('\nconfusion_matrix from Gaussian naive bayes:')
print(confusion_matrix(y_test, gausianNB_predicted ) )
accuracy = accuracy_score(y_test, gausianNB_predicted)
print('accuracy = ' + str(accuracy))
imps = permutation_importance(model2, x_test, y_test)
print("gaussinaNB feature importance:")
print(imps.importances_mean)


################## 3 ####################
# now drop WindSpeed3pm attribute
print("\n\nDropping WindSpeed3pm attribute")
X = X.drop('WindSpeed3pm', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y,  test_size=0.25, random_state=2)
print(X)
print('x_test = ' + str(x_test))
print('y_test = ' + str(y_test))

print("\n\nDecision Tree:")
dtree2 = tree.DecisionTreeClassifier(criterion="gini")
dtree2 = dtree2.fit(x_train, y_train)
y_predicted = dtree2.predict(x_test)
print('DecisionTree confusion matrix:')
print(confusion_matrix( y_test, y_predicted))
importance = dtree2.feature_importances_
print("decision tree dtree2 feature importance:")
for i, v in enumerate(importance):
	print('Feature: %0d, FName: %15s, Score: %.5f' % (i, X.columns[i], v))

print("\n\nGausianNB:")
model2 = GaussianNB()
model2.fit(x_train, y_train)
gausianNB_predicted = model2.predict(x_test)
print('\nconfusion_matrix from Gaussian naive bayes:')
print(confusion_matrix(y_test, gausianNB_predicted ) )
accuracy = accuracy_score(y_test, gausianNB_predicted)
print('accuracy = ' + str(accuracy))
imps = permutation_importance(model2, x_test, y_test)
print("gaussinaNB feature importance:")
print(imps.importances_mean)


################## 4 (dropping a group of 3) ####################
# now dropping Maxtemp, WindDir9am, and WindSpeed9am attribute
print("\n\nDropping Maxtemp, WindDir9am, and WindSpeed9am attribute")
X = X.drop('MaxTemp', axis=1)
X = X.drop('WindDir9am', axis=1)
X = X.drop('WindSpeed9am', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y,  test_size=0.25, random_state=2)
print(X)
print('x_test = ' + str(x_test))
print('y_test = ' + str(y_test))

print("\n\nDecision Tree:")
dtree2 = tree.DecisionTreeClassifier(criterion="gini")
dtree2 = dtree2.fit(x_train, y_train)
y_predicted = dtree2.predict(x_test)
print('DecisionTree confusion matrix:')
print(confusion_matrix( y_test, y_predicted))
importance = dtree2.feature_importances_
print("decision tree dtree2 feature importance:")
for i, v in enumerate(importance):
	print('Feature: %0d, FName: %15s, Score: %.5f' % (i, X.columns[i], v))

print("\n\nGausianNB:")
model2 = GaussianNB()
model2.fit(x_train, y_train)
gausianNB_predicted = model2.predict(x_test)
print('\nconfusion_matrix from Gaussian naive bayes:')
print(confusion_matrix(y_test, gausianNB_predicted ) )
accuracy = accuracy_score(y_test, gausianNB_predicted)
print('accuracy = ' + str(accuracy))
imps = permutation_importance(model2, x_test, y_test)
print("gaussinaNB feature importance:")
print(imps.importances_mean)


################## 5 (dropping a group of 3) ####################
# now dropping WindDir3pm, WindGustDir, and Location attribute
print("\n\nDropping WindDir3pm, WindGustDir, and Location attribute")
X = X.drop('WindDir3pm', axis=1)
X = X.drop('WindGustDir', axis=1)
X = X.drop('Location', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y,  test_size=0.25, random_state=2)
print(X)
print('x_test = ' + str(x_test))
print('y_test = ' + str(y_test))

print("\n\nDecision Tree:")
dtree2 = tree.DecisionTreeClassifier(criterion="gini")
dtree2 = dtree2.fit(x_train, y_train)
y_predicted = dtree2.predict(x_test)
print('DecisionTree confusion matrix:')
print(confusion_matrix( y_test, y_predicted))
importance = dtree2.feature_importances_
print("decision tree dtree2 feature importance:")
for i, v in enumerate(importance):
	print('Feature: %0d, FName: %15s, Score: %.5f' % (i, X.columns[i], v))

print("\n\nGausianNB:")
model2 = GaussianNB()
model2.fit(x_train, y_train)
gausianNB_predicted = model2.predict(x_test)
print('\nconfusion_matrix from Gaussian naive bayes:')
print(confusion_matrix(y_test, gausianNB_predicted ) )
accuracy = accuracy_score(y_test, gausianNB_predicted)
print('accuracy = ' + str(accuracy))
imps = permutation_importance(model2, x_test, y_test)
print("gaussinaNB feature importance:")
print(imps.importances_mean)