import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("Sample.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(df.head())

sns.scatterplot(x=X.iloc[:, 0], y=y)
plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predicted Values:", predictions)
print("Actual Values:", y_test.values)

print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))