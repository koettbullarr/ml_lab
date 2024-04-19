# ml_lab

Implementation of Machine Learning Algorithms from scratch for the purpose of practicing and learning.

## Machine Learning Algorithms included:

- **Linear Regression**
- **Logistic Regression**
- **K-Nearest-Neighbors**
- **Naive Bayes**
- **coming soon :)**

## Folders

- **ml_models**: Folder where the ml_lab.py is stored.
- **model_tests**: Folder where the tests for the models are made.

## Installation

Clone the repository using:

```bash
git clone https://github.com/koettbullarr/ml_lab.git
```

## Usage

Import and use the models in your Python projects:

```python
from ml_lab import LinearRegression
import evaluator

# Example usage of the Linear Regression model
model = LinearRegression(lr = 0.01, n_iters = 1000)
model.train(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate 
print(evaluator.evaluate_regression(y_test, predictions))

```


## License

This project is licensed under the MIT License - see the LICENSE file for details.
