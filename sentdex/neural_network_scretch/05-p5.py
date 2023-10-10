import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# np.random.seed(0)

# X = np.array([[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]])

X, y = spiral_data(100, 3)


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

        self.output = np.array([])

    def forward(self, inputs: np.ndarray):
        self.output = np.dot(inputs, self.weights) + self.bias


class ActivationReLU:
    def __init__(self):
        self.output = []

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


print(X.shape)
layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()

layer1.forward(X)
activation1.forward(layer1.output)

print(activation1.output)
