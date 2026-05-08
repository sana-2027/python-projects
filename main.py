import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("student_scores.csv")

X = data[["Hours"]]
y = data["Scores"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict([[5]])

print("Predicted Score:", prediction[0])

plt.scatter(data["Hours"], data["Scores"])
plt.plot(data["Hours"], model.predict(X), color="red")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Student Score Prediction")
plt.show()