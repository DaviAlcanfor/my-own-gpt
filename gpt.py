import numpy as np


class EmbeddingLayer:
  def __init__(self, vocab_size, embedding_dim):
    self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.02
    self.inputs = None

  def forward(self, inputs):
    self.inputs = inputs
    return self.embeddings[inputs]
  
