import os
import numpy as np
from PIL import Image
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from evaluate_model import evaluate_model

sys.path.append("../")

from framework import (
    FlatteningLayer,
    SoftmaxLayer,
    InputLayer,
    ReLULayer,
    ConvolutionalLayer,
    CrossEntropy,
    MaxPoolLayer,
    FullyConnectedLayer,
)


np.random.seed(42)

base_dir = "../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration"
color_train_dir = os.path.join(base_dir, "processed", "gray", "train")
color_test_dir = os.path.join(base_dir, "processed", "gray", "test")
ARCHITECTURE_NAME = 'cnn_gray'
RESULTS_DIR = '../result'

classes = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion",
]
num_classes = len(classes)


def load_data(directory):
    X, y = [], []
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).convert("L")
            img_array = np.array(img, dtype=np.float32) / 255.0
            X.append(img_array)
            label = np.zeros(num_classes, dtype=np.float32)
            label[class_idx] = 1.0
            y.append(label)
    X = np.stack(X)[..., np.newaxis]
    y = np.stack(y)
    X = (X - np.mean(X, axis=(0, 1, 2), keepdims=True)) / np.std(
        X, axis=(0, 1, 2), keepdims=True
    )
    return X, y


X_train, y_train = load_data(color_train_dir)
X_test, y_test = load_data(color_test_dir)
print(f"Train data shape: {X_train.shape}, Labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")


class CNN:
    def __init__(self, input_shape, num_classes, num_channels=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.layers = self.build_model()
        self.loss_fn = CrossEntropy()

    def build_model(self):
        H, W, C = self.input_shape
        layers = [
            InputLayer(),
            ConvolutionalLayer(kernel_size=(3, 3), num_channels=C, num_kernels=1),
            ReLULayer(),
            MaxPoolLayer(pool_size=2, stride=2),
            FlatteningLayer(),
            ReLULayer(),
            FullyConnectedLayer(sizeIn=1 * 15 * 15, sizeOut=self.num_classes),
            SoftmaxLayer(),
        ]
        return layers

    def forward(self, X):
        data = X
        for layer in self.layers:
            data = layer.forward(data)
        return data

    def compute_loss_and_gradients(self, X, Y):
        Yhat = self.forward(X)
        loss = self.loss_fn.eval(Y, Yhat)
        grad = self.loss_fn.gradient(Y, Yhat)
        grad = np.clip(grad, -10, 10)
        return loss, grad

    def backward(self, grad):
        current_grad = grad
        gradients_for_trainable = []
        for layer in reversed(self.layers):
            if isinstance(layer, (ConvolutionalLayer, FullyConnectedLayer)):
                gradients_for_trainable.append(current_grad)
            current_grad = layer.backward(current_grad)
        return gradients_for_trainable[::-1]

    def update_weights(self, learning_rate, gradients):
        grad_idx = 0
        for layer in self.layers:
            if isinstance(layer, (ConvolutionalLayer, FullyConnectedLayer)):
                if grad_idx < len(gradients):
                    if isinstance(layer, ConvolutionalLayer):
                        layer.updateKernels(learning_rate)
                    else:
                        layer.updateWeights(
                            gradients[grad_idx], learning_rate, momentum=0.9
                        )
                    grad_idx += 1

    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32, learning_rate=0.05):
        num_samples = X_train.shape[0]

        history = {
            'train_loss': [],
            'test_loss': [],
            'train_accuracy': [],
            'test_accuracy': []
        }

        for epoch in range(epochs):
            start_time = time.time()
            indices = np.random.permutation(num_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            train_loss = 0
            train_correct = 0

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                y_pred = self.forward(X_batch)
                batch_loss, grad = self.compute_loss_and_gradients(X_batch, y_batch)
                train_loss += batch_loss * (end_idx - start_idx)
                train_correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))

                gradients = self.backward(grad)
                self.update_weights(learning_rate, gradients)

            train_loss /= num_samples
            train_accuracy = train_correct / num_samples

            y_test_pred = self.forward(X_test)
            test_loss = self.loss_fn.eval(y_test, y_test_pred)
            test_accuracy = np.mean(np.argmax(y_test_pred, axis=1) == np.argmax(y_test, axis=1))

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_accuracy'].append(train_accuracy)
            history['test_accuracy'].append(test_accuracy)

            epoch_time = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )

        return history

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

if __name__ == "__main__":
    input_shape = X_train.shape[1:]
    cnn = CNN(input_shape=input_shape, num_classes=num_classes, num_channels=1)

    history = cnn.train(
        X_train, y_train, X_test, y_test, epochs=50, batch_size=32, learning_rate=0.01
    )

    true_labels = np.argmax(y_test, axis=1)
    predicted_labels = cnn.predict(X_test)
    predicted_probs = cnn.forward(X_test)

    evaluate_model(
        true_labels,
        predicted_labels,
        predicted_probs,
        classes,
        history,
        architecture_name=ARCHITECTURE_NAME,
        base_directory=RESULTS_DIR
    )
