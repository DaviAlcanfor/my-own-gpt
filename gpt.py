from abc import ABC, abstractmethod
import numpy as np


class GPTLayer(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass


class EmbeddingLayer(GPTLayer):
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.02
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.embeddings[inputs]

    def backward(self):
        pass


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


class CrossEntropyWithSoftmax(GPTLayer):
    def __init__(self):
        self.lse = None
        self.logits_flat = None
        self.labels_flat = None

    def forward(self, logits, labels):
        assert len(logits.shape) == 3
        B, T, V = logits.shape

        logits_flat = np.reshape(logits, (B * T, V))
        labels_flat = np.reshape(labels, (B * T))

        max_float = np.max(logits_flat, axis=-1, keepdims=True)
        diff = logits_flat - max_float

        lse = max_float + np.log(np.sum(np.exp(diff), axis=-1, keepdims=True))
        log_softmax = np.expand_dims(logits_flat[np.arange(B * T), labels_flat], axis=-1) - lse

        self.lse = lse
        self.logits_flat = logits_flat
        self.labels_flat = labels_flat
        self.B = B
        self.T = T
        self.V = V

        return -np.sum(log_softmax) / logits_flat.shape[0]

    def backward(self):
        pass
