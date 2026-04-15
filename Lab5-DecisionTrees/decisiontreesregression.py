from sklearn.datasets import  load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score
import numpy as np

diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Basic Decision Tree

decisiontree = DecisionTreeRegressor(random_state=42)
decisiontree.fit(xtrain, ytrain)

print("Decision Tree")
print("Train R2: ", r2_score(ytrain, decisiontree.predict(xtrain)))
print("Test R2: ", r2_score(ytest, decisiontree.predict(xtest)))

path = decisiontree.cost_complexity_pruning_path(xtrain, ytrain)
ccp_alphas = path.ccp_alphas

bestr2 = -np.inf
bestalpha = 0

for alpha in ccp_alphas:
    regression = DecisionTreeRegressor(random_state=42, ccp_alpha=alpha)
    regression.fit(xtrain, ytrain)

    r2 = r2_score(ytest, regression.predict(xtest))

    if r2 > bestr2:
        bestr2 = r2
        bestalpha = alpha

prunedregression = DecisionTreeRegressor(random_state=42, ccp_alpha=bestalpha)
prunedregression.fit(xtrain, ytrain)

print("\nPruned Decision Tree\n")
print("Best alpha: ", bestalpha)
print("Train R2: ", r2_score(ytrain, prunedregression.predict(xtrain)))
print("Test R2: ", r2_score(ytest, prunedregression.predict(xtest)))

# Random Forest

randomforestregression = RandomForestRegressor(random_state=42, n_estimators=100)
randomforestregression.fit(xtrain, ytrain)

print("\nRandom Forest Regression\n")
print("Train R2: ", r2_score(ytrain, randomforestregression.predict(xtrain)))
print("Test R2: ", r2_score(ytest, randomforestregression.predict(xtest)))

# Ada-Boost

adaboostregression = AdaBoostRegressor(n_estimators=100, random_state=42)
adaboostregression.fit(xtrain, ytrain)

print("\nAda-Boost Regression")
print("Train R2: ", r2_score(ytrain, adaboostregression.predict(xtrain)))
print("Test R2: ", r2_score(ytest, adaboostregression.predict(xtest)))