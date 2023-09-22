# ur = 0.858116442440411
# r = 0.847793015902031
# q = 6
# n = 472
# k = 12
# F = (ur - r)/(1 - ur)*(n-k-1)/q
# print(F);


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#regression
import statsmodels.api as sm
from statsmodels.formula.api import ols
# import seaborn as sn
pd.set_option("display.max_columns", None)

mergedData_tree = pd.read_csv("Book3.csv")

# SUGGESTIONS FROM INSTRUCTORS
# log(population):
# mergedData_tree['Pop_2021'] = np.log(mergedData_tree['Pop_2021'])
# does not work, coef of Pop_2021 becomes negative, R^2 becomes 0.69

# num of innouts / population:
# mergedData_tree['#_in-n-outs'] = mergedData_tree['#_in-n-outs'] / mergedData_tree['Pop_2021']
# does not work at all, R^2 = 0.154

# the reason of such drastic change might be because Linear Regression is just not the suitable model
# which is what Prof. Dubin thinks (Decision tree might be your best model)

mergedData = mergedData_tree.drop('innout_tree', axis=1)
TN_data = pd.read_csv("TN_data.csv")

# print(mergedData_tree.isnull().sum())
# print(mergedData)

# --------------------------------------------------------------------------------------------------------------
# Poisson Regression
#
# from sklearn.linear_model import PoissonRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
#
# y = mergedData['#_in-n-outs']
# X = mergedData.drop('#_in-n-outs', axis=1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# poisson = PoissonRegressor()
# poisson.fit(X_train, y_train)
# r_squared = poisson.score(X_test, y_test)
# print("Score of Poisson: ", r_squared)
# y_pred = poisson.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print('Mean squared error: ', mse)

# ---------------------------------------------------------------------------
# Linear Regression

# # only innout of CA
# mergedData_tree = pd.read_csv("Book7.csv")
# mergedData = mergedData_tree.drop('innout_tree', axis=1)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X = mergedData.drop('#_in-n-outs', axis=1)
y = mergedData['#_in-n-outs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
r_squared = lr.score(X_test, y_test)
print()
print("Score for Linear Regression: ", r_squared)

y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error: ', mse)
#
# TN_outcome = lr.predict(TN_data)
# for i in TN_outcome:
#     print(i);

# ---------------------------------------------------------------------------
#Linear Regression with statsmodels


X = mergedData.drop('#_in-n-outs', axis=1)
y = mergedData['#_in-n-outs']
lm_fit_linear1 = sm.OLS(y, sm.add_constant(X), missing='drop').fit()
print(lm_fit_linear1.summary())

# ---------------------------------------------------------------------------
# Plot
# plt.scatter(x=mergedData["#_cars_capita"], y=mergedData["#_in-n-outs"])
# plt.title("#_in-n-outs and #_cars_capita")
# plt.xlabel("#_cars_capita")
# plt.ylabel("#_in-n-outs")
# plt.show()

# ---------------------------------------------------------------------------
# Subplots
# myFig = plt.figure()
# myAx1 = myFig.add_subplot(2, 2, 1)
# myAx2 = myFig.add_subplot(2, 2, 2)
# myAx3 = myFig.add_subplot(2, 2, 3)
# myAx4 = myFig.add_subplot(2, 2, 4)
# # myAx5 = myFig.add_subplot(3, 3, 5)
# # myAx6 = myFig.add_subplot(3, 3, 6)
# # myAx7 = myFig.add_subplot(3, 3, 7)
# # myAx8 = myFig.add_subplot(3, 3, 8)
# # myAx9 = myFig.add_subplot(3, 3, 9)
# #
# myAx1.scatter(x=mergedData["Pop_2021"], y=mergedData["#_in-n-outs"])
# # myAx1.set_title("#_in-n-outs and Pop_2021")
# myAx1.set_xlabel("Pop_2021")
# myAx1.set_ylabel("#_in-n-outs")
#
# myAx2.scatter(x=mergedData["total_mileage_of_freeways"], y=mergedData["#_in-n-outs"])
# # myAx2.set_title("#_in-n-outs and total_mileage_of_freeways")
# myAx2.set_xlabel("total_mileage_of_freeways")
# myAx2.set_ylabel("#_in-n-outs")
#
# myAx3.scatter(x=mergedData["GDP_of_State"], y=mergedData["#_in-n-outs"])
# # myAx3.set_title("#_in-n-outs and densityMi")
# myAx3.set_xlabel("GDP_of_State")
# myAx3.set_ylabel("#_in-n-outs")
#
# myAx4.scatter(x=mergedData["#_BurgerKing"], y=mergedData["#_in-n-outs"])
# # myAx4.set_title("#_in-n-outs and growthSince2010")
# myAx4.set_xlabel("#_BurgerKing")
# myAx4.set_ylabel("#_in-n-outs")
#
# myAx5.scatter(x=mergedData["#_BurgerKing"], y=mergedData["#_in-n-outs"])
# # myAx5.set_title("#_in-n-outs and #_BurgerKing")
# myAx5.set_xlabel("#_BurgerKing")
# myAx5.set_ylabel("#_in-n-outs")
#
# myAx6.scatter(x=mergedData["median_income_capita"], y=mergedData["#_in-n-outs"])
# # myAx6.set_title("#_in-n-outs and median_income_capita")
# myAx6.set_xlabel("median_income_capita")
# myAx6.set_ylabel("#_in-n-outs")
#
# myAx7.scatter(x=mergedData["GDP_of_State"], y=mergedData["#_in-n-outs"])
# # myAx7.set_title("#_in-n-outs and GDP_of_State")
# myAx7.set_xlabel("GDP_of_State")
# myAx7.set_ylabel("#_in-n-outs")
#
# myAx8.scatter(x=mergedData["#_innout_state"], y=mergedData["#_in-n-outs"])
# # myAx8.set_title("#_in-n-outs and #_innout_state")
# myAx8.set_xlabel("#_innout_state")
# myAx8.set_ylabel("#_in-n-outs")
#
# myAx9.scatter(x=mergedData["#_cars_capita"], y=mergedData["#_in-n-outs"])
# # myAx9.set_title("#_in-n-outs and Percent_Change_GDP")
# myAx9.set_xlabel("#_cars_capita")
# myAx9.set_ylabel("#_in-n-outs")
# plt.subplots_adjust(wspace=0.8, hspace=0.8)
# plt.show()

# ---------------------------------------------------------------------------

# Decision Tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

y = mergedData_tree['innout_tree']
X = mergedData_tree.drop('#_in-n-outs', axis=1)
X = X.drop('innout_tree', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.7, stratify=y)

model_tree = DecisionTreeClassifier(criterion='gini', random_state=42)
model_tree.fit(X_train, y_train)

y_pred = model_tree.predict(X_test)
# print(y_pred)

print('accuracy:', model_tree.score(X_test, y_test))
print(classification_report(y_test, y_pred))
#
# plot_tree(model_tree, filled=True)
# plt.savefig('decision_tree.png')
#
# feat_import = model_tree.feature_importances_
# print('feature importances:\n', X.columns.values, '\n', feat_import)
#
# TN_outcome = model_tree.predict(TN_data)
# for i in TN_outcome:
#     print(i);
#
#
# #Linear Regression with statsmodel
# mergedData = pd.read_csv("have_innout.csv")
# X = mergedData.drop('#_in-n-outs', axis=1)
# y = mergedData['#_in-n-outs']
# lm_fit_linear2 = sm.OLS(y, sm.add_constant(X), missing='drop').fit()
# print(lm_fit_linear2.summary())
# # print(X.corr())
#
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# lr = LinearRegression()
# lr.fit(X_train, y_train)
# r_squared = lr.score(X_test, y_test)
# print()
# print("Score for Linear Regression: ", r_squared)
#
# y_pred = lr.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print('Mean squared error: ', mse)
#
# TN_outcome = lr.predict(TN_data)
# for i in TN_outcome:
#     print(i);