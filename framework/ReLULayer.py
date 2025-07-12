from .Layer import Layer
import numpy as np

class ReLULayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        output = np.maximum(0, dataIn)
        self.setPrevOut(output)
        return output

    def gradient(self):
        return (self.getPrevIn() >= 0).astype(np.float32)

    def backward(self, gradIn):
        return gradIn * self.gradient()
