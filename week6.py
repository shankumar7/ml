import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Sample.csv")

print("Dataset Length:", len(df))
print("Dataset Shape:", df.shape)
print("Dataset:")
print(df.head())
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


model_gini = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

model_gini.fit(X_train, y_train)


print("\nUsing Gini Index")

plt.figure(figsize=(16, 8))
plot_tree(
    model_gini,
    feature_names=X.columns,
    class_names=[str(c) for c in model_gini.classes_],
    filled=True,
    rounded=True
)
plt.title("Decision Tree Using Gini Index")
plt.show()


y_pred_gini = model_gini.predict(X_test)

print("\nResults Using Gini Index:")
print("Predicted Values:\n", y_pred_gini)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gini))
print("Accuracy:", accuracy_score(y_test, y_pred_gini) * 100)
print("Report:\n", classification_report(
    y_test, y_pred_gini, zero_division=0
))


model_entropy = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

model_entropy.fit(X_train, y_train)

print("\nUsing Entropy")

plt.figure(figsize=(16, 8))
plot_tree(
    model_entropy,
    feature_names=X.columns,
    class_names=[str(c) for c in model_entropy.classes_],
    filled=True,
    rounded=True
)
plt.title("Decision Tree Using Entropy")
plt.show()

y_pred_entropy = model_entropy.predict(X_test)

print("\nResults Using Entropy:")
print("Predicted Values:\n", y_pred_entropy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_entropy))
print("Accuracy:", accuracy_score(y_test, y_pred_entropy) * 100)
print("Report:\n", classification_report(
    y_test, y_pred_entropy, zero_division=0
))