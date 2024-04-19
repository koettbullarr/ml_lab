import numpy as np
from collections import Counter

# LINEAR REGRESSION
class LinearRegression:
    
    def __init__(self, lr = 0.001, n_iters = 1000):
        
        self.lr = lr
        self.n_iters = n_iters
        
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        
        # Initialize the weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Loop in range of numbers of iterations
        for _ in range(self.n_iters):
        
            # Calculate the prediction value
            y_pred = np.dot(X , self.weights) + self.bias
            
            # Calculate the gradients of weights and bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update weights and bias
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# LOGISTIC REGRESSION

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    
    def __init__(self, lr = 0.001, n_iters = 1000):
        
        self.lr = lr
        self.n_iters = n_iters
        
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            
            y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
            
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
        
    def predict(self, X):
        y_pred =  sigmoid(np.dot(X, self.weights) + self.bias)
        return [0 if y <= 0.5 else 1 for y in y_pred]
        
# KNN
   
# Define the euclidean disctance calculator 
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    
    def __init__(self, k: int):
        
        self.k = k
        
    def fit(self, X, y):
        
        # Initialize variables
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        
        # Predict clusters with the help of our helper function
        return [self._predict(x) for x in X]


        
    def _predict(self, x):
        
        # Compute the eucledian distance
        distances = [euclidean_distance(x, X_train) for X_train in self.X_train]
        
        # Get the closest k neighbors by distances
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Majority votes
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
        
# NAIVE BAYES

class NaiveBayes:
    
    def __init__(self):
        # No init function because no params
        pass    
        
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        self._mean = np.zeros((n_classes, n_features), dtype = np.float64)
        self._var = np.zeros((n_classes, n_features), dtype = np.float64)
        self._priors = np.zeros(n_classes, dtype = np.float64)
        
        for i, c in enumerate(self._classes):
            
            X_c = X[y == c]
            
            self._mean[i, :] = X_c.mean(axis=0)
            self._var[i, :] = X_c.var(axis=0)
            self._priors[i] = X_c.shape[0] / float(n_samples)
            
    def predict(self, X):
        
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        
        posteriors = []
        
        for i, c in enumerate(self._classes):
            
            prior = np.log(self._priors[i])
            posterior = np.sum(np.log(self._pdf(i, x)))
            posterior += prior
            posteriors.append(posterior)
            
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_i, x):
        
        mean = self._mean[class_i]
        var = self._var[class_i]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
            
        
        
        
            
            
        
        
        
        
        
    
        
        
        
        