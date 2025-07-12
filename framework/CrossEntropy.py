import numpy as np

class CrossEntropy:
    def eval(self, Y, Yhat, epsilon=1e-7):
        Yhat = np.clip(Yhat, epsilon, 1. - epsilon)
        return -np.mean(np.sum(Y * np.log(Yhat), axis=1))
    
    def gradient(self, Y, Yhat):
        return Yhat - Y