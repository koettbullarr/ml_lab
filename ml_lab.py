import numpy as np

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
    

        
        
    
        
        
        
        