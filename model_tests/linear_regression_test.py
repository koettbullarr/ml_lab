from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import evaluator
from ml_models.ml_lab import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=123)

plt.figure(figsize=(8,4))
plt.scatter(X, y, color='black', marker='o')
plt.show()

lm = LinearRegression()
lm.fit(X_train,y_train)
pred = lm.predict(X_test)
print(evaluator.evaluate_regression(y_test, pred))

plt.figure(figsize=(8,4))
plt.scatter(X, y, color='black', marker='o')
plt.plot(X, lm.predict(X))
plt.show()
