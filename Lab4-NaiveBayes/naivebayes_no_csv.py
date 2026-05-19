from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load Iris dataset directly
iris = load_iris()

# Features and target
x = iris.data
y = iris.target

# Split dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create and train model
model = GaussianNB()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

print("\nPrediction Results\n")

correct = 0
wrong = 0

for i in range(len(y_test)):

    actual = iris.target_names[y_test[i]]
    predicted = iris.target_names[y_pred[i]]

    print(f"Sample {i+1}:")
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}", end=" ")

    if y_test[i] == y_pred[i]:
        print("-> Correct\n")
        correct += 1
    else:
        print("-> Wrong\n")
        wrong += 1

print("Summary:")
print(f"Correct Predictions: {correct}")
print(f"Wrong Predictions: {wrong}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")