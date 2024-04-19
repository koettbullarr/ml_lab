from sklearn.model_selection import train_test_split
from sklearn import datasets
from ml_models.ml_lab import LogisticRegression
import evaluator

df = datasets.load_breast_cancer()
X, y = df.data, df.target 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=123)

lr = LogisticRegression(lr = 0.01, n_iters=1000)
lr.fit(X_train, y_train)
preds = lr.predict(X_test)

print(evaluator.evaluate_classification(y_test, preds))