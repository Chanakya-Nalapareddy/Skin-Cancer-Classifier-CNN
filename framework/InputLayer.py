from .Layer import Layer
import numpy as np


class InputLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        if dataIn is None or not isinstance(dataIn, np.ndarray):
            raise ValueError("InputLayer expects a non-None NumPy array.")
        self.setPrevIn(dataIn)
        self.setPrevOut(dataIn)
        return dataIn

    def gradient(self):
        pass

    def backward(self, gradIn): 
        return gradIn
