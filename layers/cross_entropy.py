import numpy as np

from layers.base import GPTLayer


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
