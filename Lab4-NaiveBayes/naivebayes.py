import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("iris.csv")

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = GaussianNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("\n Prediction Results\n")

correct = 0
wrong = 0

for i in range(len(y_test)):
    actual = label_encoder.inverse_transform([y_test[i]])[0]
    predicted = label_encoder.inverse_transform([y_pred[i]])[0]

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