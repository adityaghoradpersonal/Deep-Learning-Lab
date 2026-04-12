from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

x = iris.data
y = iris.target
class_names = iris.target_names

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
accuracy_results = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors = k, metric = 'euclidean')
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    accuracy_results.append((k, acc))

best_k = sorted(accuracy_results, key=lambda x: (-x[1], x[0]))[0][0]

for k, acc in accuracy_results:
    print(f"k = {k}, accuracy = {acc:.4f}")

print(f"\nBest K selected: {best_k}")

final_model = KNeighborsClassifier(n_neighbors = best_k, metric='euclidean')
final_model.fit(x_train, y_train)

y_pred = final_model.predict(x_test)

print("\nDetailed Prediction Results:\n")

correct = 0
wrong = 0

for i in range(len(y_test)):
    actual = class_names[y_test[i]]
    predicted = class_names[y_pred[i]]
    
    print(f"Sample {i+1}:")
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}", end=" ")
    
    if y_test[i] == y_pred[i]:
        print("→ Correct\n")
        correct += 1
    else:
        print("→ Wrong\n")
        wrong += 1

print("Summary:")
print(f"Correct Predictions: {correct}")
print(f"Wrong Predictions: {wrong}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")