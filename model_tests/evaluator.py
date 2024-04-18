from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_regression(y_true, y_pred):
    return f"MAE: {mean_absolute_error(y_true, y_pred)}\nMSE: {mean_squared_error(y_true, y_pred)}\nRMSE: {np.sqrt(mean_squared_error(y_true, y_pred))}"

def evaluate_classification(y_true, y_pred):
    return f"{classification_report(y_true, y_pred)}\n{confusion_matrix(y_true, y_pred)}"