import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
df=pd.read_csv("Sample.csv")

X=df.iloc[:,:-1]

model=KMeans(n_clusters=3,random_state=42)
model.fit(X)
labels=model.labels_

print("Cluster Labels: \n",labels)
print("Cluster Centers: \n",model.cluster_centers_)

plt.scatter(X.iloc[:,0],X.il0c[:,1],c=labels)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

