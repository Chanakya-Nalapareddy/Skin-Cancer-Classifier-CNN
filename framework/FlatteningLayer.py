import numpy as np
from .Layer import Layer


class FlatteningLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        if dataIn.ndim != 4:
            raise ValueError(f"Expected 4D input (N, H, W, C), got {dataIn.ndim}D.")
        self.setPrevIn(dataIn)
        flattened = dataIn.reshape(dataIn.shape[0], -1) 
        self.setPrevOut(flattened)
        return flattened

    def gradient(self):
        return np.eye(self.getPrevOut().shape[1])

    def backward(self, gradIn):
        gradOut = super().backward(gradIn)
        return gradOut.reshape(self.getPrevIn().shape)
