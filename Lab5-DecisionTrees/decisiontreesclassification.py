from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Basic Decision Tree

decisiontree = DecisionTreeClassifier(random_state = 42)
decisiontree.fit(x_train, y_train)

y_train_pred = decisiontree.predict(x_train)
y_test_pred = decisiontree.predict(x_test)

print("Decision Tree Classification")
print(f"Train accuracy: {accuracy_score(y_train, y_train_pred)}")
print(f"Test accuracy: {accuracy_score(y_test, y_test_pred)}")

# Cost Complexity Pruning

path = decisiontree.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas = path.ccp_alphas

best_acc = 0
best_alpha = 0

for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state = 42, ccp_alpha = alpha)
    clf.fit(x_train, y_train)

    y_test_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_test_pred)

    if acc > best_acc:
        best_acc = acc
        best_alpha = alpha

pruned_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
pruned_clf.fit(x_train, y_train)

print("\nPruned Decision Tree\n")
print(f"Best alpha: {best_alpha}")
print(f"Train accuracy: {accuracy_score(y_train, pruned_clf.predict(x_train))}")
print(f"Test accuracy: {accuracy_score(y_test, pruned_clf.predict(x_test))}")

# Random Forest

randomforestclassifier = RandomForestClassifier(n_estimators=100, random_state=42)
randomforestclassifier.fit(x_train, y_train)

train_acc = accuracy_score(y_train, randomforestclassifier.predict(x_train))
test_acc = accuracy_score(y_test, randomforestclassifier.predict(x_test))

print("\nRandom Forest\n")
print(f"Train accuracy: {train_acc}")
print(f"Test accuracy: {test_acc}")

# AdaBoost

adaboostclassifer = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboostclassifer.fit(x_train, y_train)

print("\nAda-Boost\n")
print(f"Train accuracy: {accuracy_score(y_train, adaboostclassifer.predict(x_train))}")
print(f"Test accuracy: {accuracy_score(y_test, adaboostclassifer.predict(x_test))}")