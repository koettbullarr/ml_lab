import numpy as np
from collections import Counter

class LinearRegression:
    
    def __init__(self, lr = 0.01, n_iters = 1000):
        
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
        

        
        
    
        
        
        
        