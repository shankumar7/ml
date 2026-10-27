import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics

df=pd.read_csv("sample.csv")
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

model=LogisticRegression(solver="liblinear")
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

cm=metrics.confusion_matrix(y_test,y_pred)
print(cm)

sns.heatmap(pd.DataFrame(cm), annot=True , fmt="g",color="YlGnBu")
plt.title("Confusion Matrix")
plt.xlabel("predicted")
plt.ylabel("Actual")
plt.show()

print("acuuracy: ",metrics.accuracy_score(y_test,))