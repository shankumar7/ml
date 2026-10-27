import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Sample.csv")

X = df.iloc[:, 0:1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Values:", y_pred)
print("Actual Values:", y_test.values)

plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, model.predict(X_train), color="blue")
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.title("Simple Linear Regression")
plt.show()

plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, model.predict(X_train), color="blue")
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.title("Testing Set")
plt.show()