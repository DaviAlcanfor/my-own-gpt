import numpy as np

from base import GPTLayer
from softmax import Softmax
from linear import Linear

class MultiHeadAttention(GPTLayer):
    def __init__(self, dim_in, dim_out, n_heads):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_heads = n_heads

        self.query_linear = Linear(dim_in, dim_out, include_bias=False)
        self.key_linear = Linear(dim_in, dim_out, include_bias=False)
        self.value_linear = Linear(dim_in, dim_out, include_bias=False)
        self.softmax = Softmax()
        self.final_projection = Linear(dim_out, dim_out, include_bias=False)

        # Save for backprop
        self.queries_matrix = None
        self.keys_matrix = None
        self.values_matrix = None
        self.attention_weights = None

    def forward(self, inputs, mask = None):
        B, T, D = inputs.shape

        queries = self.query_linear.forward(inputs)
        keys = self.key_linear.forward(inputs)
        values = self.value_linear.forward(inputs)

        # (B, token_length, number_heads, head_dimension)
        queries = np.reshape(queries, (B, T, self.n_heads, self.dim_out // self.n_heads))
        keys = np.reshape(keys, (B, T, self.n_heads, self.dim_out // self.n_heads))
        values = np.reshape(values, (B, T, self.n_heads, self.dim_out // self.n_heads))

        # (B, number_heads, token_length, head_dimension)
        queries = np.transpose(queries, axes=[0, 2, 1, 3])
        keys = np.transpose(keys, axes=[0, 2, 1, 3])
        values = np.transpose(values, axes=[0, 2, 1, 3])

        attention_scores = queries @ np.transpose(keys, axes=[0, 1, 3, 2])
        attention_scores = attention_scores / np.sqrt(keys.shape[-1])

        if mask is not None:
            attention_scores += mask

        attention_weights = self.softmax.forward(attention_scores)

        final_vector = attention_weights @ values

        # (B, token_length, number_heads, head_dimension)
        final_vector = np.transpose(final_vector, axes=[0, 2, 1, 3])
        final_vector = np.reshape(final_vector, (B, T, self.dim_out))
        final_vector = self.final_projection.forward(final_vector)

        # Store stuff for backprop
        self.queries_matrix = queries
        self.keys_matrix = keys
        self.values_matrix = values
        self.attention_weights = attention_weights

        return final_vector

    def backward(self):
      pass
