import numpy as np

class LayerNorm:
    def __init__(self, input_dim):
        # Parameters
        self.input_dim = input_dim
        self.gain = np.ones((1, input_dim))
        self.bias = np.zeros((1, input_dim))

        # Things to keep track of
        self.mean = None
        self.mu = None
        self.var = None
        self.std = None
        self.out = None

        # Gradients
        self.grad_gain = np.zeros(self.gain.shape)
        self.grad_bias = np.zeros(self.bias.shape)

    def forward(self, inputs):
        epsilon = 1e-5
        self.mean = (1 / self.input_dim) * np.sum(inputs, axis=-1, keepdims=True)
        self.mu = inputs - self.mean
        self.var = (1 / self.input_dim) * np.sum(np.square(self.mu), axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + epsilon)
        self.out = self.gain * (self.mu / self.std) + self.bias

        return self.out
