import numpy as np

from layers.base import GPTLayer


class EmbeddingLayer(GPTLayer):
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.02
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.embeddings[inputs]

    def backward(self):
        pass
