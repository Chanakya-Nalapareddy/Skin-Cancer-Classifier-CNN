from .Layer import Layer
import numpy as np


class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut):
        super().__init__()
        limit = np.sqrt(6 / (sizeIn + sizeOut))
        self.weights = np.random.uniform(-limit, limit, (sizeIn, sizeOut))
        self.biases = np.zeros((1, sizeOut))
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.biases)

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        if weights.shape[1] != self.weights.shape[1]:
            raise ValueError(
                f"Expected weights shape {self.weights.shape}, got {weights.shape}."
            )
        self.weights = weights

    def getBiases(self):
        return self.biases

    def setBiases(self, biases):
        if biases.shape != self.biases.shape:
            raise ValueError(
                f"Expected biases shape {self.biases.shape}, got {biases.shape}."
            )
        self.biases = biases
    
    def forward(self, dataIn):
        if dataIn.ndim != 2:
            raise ValueError(f"Expected 2D input, got {dataIn.ndim}D.")
        output = np.dot(dataIn, self.weights) + self.biases
        self.setPrevIn(dataIn)
        self.setPrevOut(output)
        return output
    
    def gradient(self, gradIn):
        dJdW = np.dot(self.getPrevIn().T, gradIn) / gradIn.shape[0]
        dJdb = np.sum(gradIn, axis=0, keepdims=True) / gradIn.shape[0]
        return dJdW, dJdb
    
    def backward(self, gradIn):
        return np.dot(gradIn, self.weights.T)
    
    def updateWeights(self, gradIn, eta, momentum=0.9):
        dJdW, dJdb = self.gradient(gradIn)
        self.velocity_w = momentum * self.velocity_w - eta * dJdW
        self.velocity_b = momentum * self.velocity_b - eta * dJdb
        self.weights += self.velocity_w
        self.biases += self.velocity_b