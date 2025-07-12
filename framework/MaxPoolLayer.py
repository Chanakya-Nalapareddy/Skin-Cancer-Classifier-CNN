import numpy as np
from .Layer import Layer

class MaxPoolLayer(Layer):
    def __init__(self, pool_size, stride):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.indices = None

    def forward(self, dataIn):
        if dataIn.ndim != 4:
            raise ValueError(f"Expected 4D input, got {dataIn.ndim}D.")
        N, H, W, C = dataIn.shape
        out_height = (H - self.pool_size) // self.stride + 1
        out_width = (W - self.pool_size) // self.stride + 1
        
        patches = np.lib.stride_tricks.as_strided(
            dataIn,
            shape=(N, out_height, out_width, self.pool_size, self.pool_size, C),
            strides=(dataIn.strides[0], self.stride * dataIn.strides[1], self.stride * dataIn.strides[2],
                     dataIn.strides[1], dataIn.strides[2], dataIn.strides[3])
        )
        output = np.max(patches, axis=(3, 4))
        self.indices = np.argmax(patches.reshape(N, out_height, out_width, -1, C), axis=3)
        self.setPrevIn(dataIn)
        self.setPrevOut(output)
        return output

    def backward(self, gradIn):
        N, out_h, out_w, C = gradIn.shape
        prevIn = self.getPrevIn()
        N, H, W, C = prevIn.shape
        gradIn_pooled = np.zeros_like(prevIn)
        
        for n in range(N):
            for c in range(C):
                idx_flat = self.indices[n, :, :, c]
                i_coords = np.arange(out_h)[:, None] * self.stride + (idx_flat // self.pool_size)
                j_coords = np.arange(out_w) * self.stride + (idx_flat % self.pool_size)
                gradIn_pooled[n, i_coords, j_coords, c] += gradIn[n, :, :, c]
        
        return gradIn_pooled
    
    def gradient(self):
        pass