import numpy as np
from .Layer import Layer

class ConvolutionalLayer(Layer):
    def __init__(self, kernel_size, num_channels=1, num_kernels=1):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.num_channels = num_channels
        self.num_kernels = num_kernels
        limit = np.sqrt(6 / (num_channels * kernel_size[0] * kernel_size[1] + num_kernels * kernel_size[0] * kernel_size[1]))
        self.kernels = np.random.uniform(-limit, limit, (num_kernels, num_channels, kernel_size[0], kernel_size[1]))
        self.velocity_k = np.zeros_like(self.kernels)
        self.input_tensor = None

    def setKernels(self, weights):
        self.kernels = weights
    
    def getKernels(self):
        return self.kernels

    def im2col(self, dataIn):
        N, H, W, C = dataIn.shape
        kh, kw = self.kernel_size
        out_h = H - kh + 1
        out_w = W - kw + 1
        col = np.zeros((N * out_h * out_w, C * kh * kw))
        idx = 0
        for n in range(N):
            for i in range(out_h):
                for j in range(out_w):
                    col[idx] = dataIn[n, i:i+kh, j:j+kw, :].ravel()
                    idx += 1
        return col, out_h, out_w

    def forward(self, dataIn):
        if dataIn.ndim != 4:
            raise ValueError(f"Expected 4D input (N, H, W, C), got {dataIn.ndim}D.")
        N, H, W, C = dataIn.shape
        if H < self.kernel_size[0] or W < self.kernel_size[1]:
            raise ValueError(f"Input size ({H}, {W}) smaller than kernel size {self.kernel_size}.")
        if C != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {C}")
        self.input_tensor = dataIn
        self.setPrevIn(dataIn)
        kh, kw = self.kernel_size
        out_h = H - kh + 1
        out_w = W - kw + 1
        
        col, out_h, out_w = self.im2col(dataIn)
        kernels_flat = np.flip(self.kernels, axis=(2, 3)).transpose(0, 2, 3, 1).reshape(self.num_kernels, -1)
        output = np.dot(col, kernels_flat.T).reshape(N, out_h, out_w, self.num_kernels)
        self.setPrevOut(output)
        return output

    def gradient(self, gradIn):
        N, out_h, out_w, K = gradIn.shape
        kh, kw = self.kernel_size
        kernel_grad = np.zeros_like(self.kernels)
        col, _, _ = self.im2col(self.input_tensor)
        grad_flat = gradIn.reshape(N * out_h * out_w, K)
        kernel_grad_flat = np.dot(grad_flat.T, col) / N
        kernel_grad = kernel_grad_flat.reshape(self.num_kernels, kh, kw, self.num_channels).transpose(0, 3, 1, 2)
        return kernel_grad

    def backward(self, gradIn):
        N, out_h, out_w, K = gradIn.shape
        kh, kw = self.kernel_size
        N, H, W, C = self.input_tensor.shape
        input_grad = np.zeros_like(self.input_tensor)
        
        self.kernel_grad = self.gradient(gradIn)
        
        grad_flat = gradIn.reshape(N * out_h * out_w, K)
        kernels_flat = np.flip(self.kernels, axis=(2, 3)).transpose(0, 2, 3, 1).reshape(K, -1)
        dX_col = np.dot(grad_flat, kernels_flat)
        idx = 0
        for n in range(N):
            for i in range(out_h):
                for j in range(out_w):
                    input_grad[n, i:i+kh, j:j+kw, :] += dX_col[idx].reshape(kh, kw, C)
                    idx += 1
        return input_grad

    def updateKernels(self, learning_rate, momentum=0.9):
        if not hasattr(self, 'kernel_grad'):
            raise ValueError("Kernel gradient not computed. Call backward first.")
        self.velocity_k = momentum * self.velocity_k - learning_rate * self.kernel_grad
        self.kernels += self.velocity_k
