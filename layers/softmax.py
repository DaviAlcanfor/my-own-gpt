import numpy as np

from layers.base import GPTLayer


class Softmax(GPTLayer):
    def __init__(self):
        self.softmax_out = None

    def forward(self, inputs):
        max_logit = np.max(inputs, axis=-1, keepdims=True)
        softmax_out = np.exp(inputs - max_logit) / np.sum(np.exp(inputs - max_logit), axis=-1, keepdims=True)
        self.softmax_out = softmax_out
        return softmax_out

    def backward(self):
        pass
