import numpy as np


np.random.seed(0)

X = np.array([[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]])


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

        self.output = np.array([])

    def forward(self, inputs: np.ndarray):
        self.output = np.dot(inputs, self.weights) + self.bias


layer1 = LayerDense(4, 5)
layer2 = LayerDense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)
