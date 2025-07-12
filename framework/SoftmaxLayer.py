from .Layer import Layer
import numpy as np

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        if dataIn.ndim != 2:
            raise ValueError(
                f"Expected 2D input (batch_size, num_classes), got {dataIn.ndim}D."
            )
        shift_data = dataIn - np.max(dataIn, axis=1, keepdims=True)
        exp_data = np.exp(shift_data)
        output = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        self.setPrevIn(dataIn)
        self.setPrevOut(output)
        return output

    def gradient(self):
        g = self.getPrevOut()
        batch_size, num_classes = g.shape
        jacobians = np.zeros((batch_size, num_classes, num_classes))

        for k in range(batch_size):
            diag_g = np.diag(g[k])
            outer_g = np.outer(g[k], g[k])
            jacobians[k] = diag_g - outer_g

        return jacobians
