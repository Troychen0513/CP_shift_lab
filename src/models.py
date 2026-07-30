"""Simple regression models for the CP experiments."""

import numpy as np

class PolyModel:
    """Fit a polynomial curve to predict the center of Y."""
    
    def __init__(self,degree:int):
        self.degree = degree
        self.coef = None
        
    def fit(self,x,y) -> None:
        """Fit the polynomial model on training data."""
        x = np.asarray(x,dtype = float)
        y = np.asarray(y,dtype = float)
        
        self.coef = np.polyfit(x,y,deg = self.degree)
        
    def predict(self,x) -> np.ndarray:
        """Predict Y values for new X."""
        
        if self.coef is None:
            raise ValueError("model must be fitted before prediction")
        
        x = np.asarray(x,dtype = float)
        return np.polyval(self.coef, x)
    
    