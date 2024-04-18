import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ml_models.ml_lab import KNN
import evaluator

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)

plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap)
plt.show()


knn = KNN(k=3)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
print(evaluator.evaluate_classification(y_test, preds))

fig, ax = plt.subplots(ncols=2)
ax[0].scatter(X[:,2], X[:,3], c=y, cmap=cmap)
ax[0].set_title("True")
ax[1].scatter(X[:,2], X[:,3], c=knn.predict(X), cmap=cmap)
ax[1].set_title("Predicted")
plt.show()