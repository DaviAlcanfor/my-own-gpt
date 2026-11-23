import numpy as np
from abc import ABC, abstractmethod


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


class Linear:
    def __init__(self, input_units, output_units, include_bias=True):
        self.weights = np.random.randn(input_units, output_units) * 0.02
        if include_bias:
            self.bias = np.zeros((1, output_units))
            self.grad_bias = np.zeros(self.bias.shape)
        self.inputs = None
        self.include_bias = include_bias

        self.grad_weights = np.zeros(self.weights.shape)

    def forward(self, inputs):
        self.inputs = inputs
        if self.include_bias:
            return inputs @ self.weights + self.bias
        else:
            return inputs @ self.weights

    def backward(self, prev_grad):
        inputs_flat = self.inputs.reshape(-1, self.inputs.shape[-1])
        prev_grad_flat = prev_grad.reshape(-1, prev_grad.shape[-1])

        dL_dA = prev_grad @ self.weights.T
        dL_dW = inputs_flat.T @ prev_grad_flat

        if self.include_bias:
            dL_dB = np.sum(prev_grad_flat, axis=0, keepdims=True)
            self.grad_bias += dL_dB

        self.grad_weights += dL_dW


class ReLU:
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, prev_grad):
        return np.maximum(0, np.sign(self.inputs)) * prev_grad


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


class FeedForwardNetwork:
    def __init__(self, input_units, hidden_units, output_units):
        self.linear1 = Linear(input_units, hidden_units)
        self.activation = ReLU()
        self.linear2 = Linear(hidden_units, output_units)

    def forward(self, inputs):
        x = self.linear1.forward(inputs)
        x = self.activation.forward(x)
        x = self.linear2.forward(x)
        return x

    def backward(self, prev_grad):
        prev_grad = self.linear2.backward(prev_grad)
        prev_grad = self.activation.backward(prev_grad)
        prev_grad = self.linear1.backward(prev_grad)
        return prev_grad

    def get_params(self):
        return {"layer_name": "FNN", "layer_parameters": None, "children": [self.linear1.get_params(), self.linear2.get_params()]}


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


class TransformerBlock:
    def __init__(self, dim_model, n_heads, ln_position="pre", res_weight_init_scale = 1):
        self.ln_position = ln_position

        self.multi_head_attention = MultiHeadAttention(dim_model, dim_model, n_heads) # Masked
        self.layernorm1 = LayerNorm(dim_model)

        self.ffn = FeedForwardNetwork(dim_model, 4 * dim_model, dim_model)
        self.layernorm2 = LayerNorm(dim_model)

    def get_params(self): ...

    def forward(self, inputs, mask_attention = None, padding_mask = None):
        if self.ln_position == "post":
            out = inputs + self.multi_head_attention.forward(inputs, mask=mask_attention)
            out = self.layernorm1.forward(out)

            out = out + self.ffn.forward(out)
            out = self.layernorm2.forward(out)

        elif self.ln_position == "pre":
            out = inputs + self.multi_head_attention.forward(self.layernorm1.forward(inputs), mask=mask_attention)
            out = out + self.ffn.forward(self.layernorm2.forward(out))

        return out

    def backward(self, prev_grad): ...


class GPT:
    def __init__(self, vocab_size, blocks, dim_model, n_heads, ln_position="pre"):
        self.vocab_size = vocab_size

        # Decoder
        self.embedding_decoder = EmbeddingLayer(vocab_size, embedding_dim=dim_model)
        self.positional_encodings_decoder = PositionalEncodings(dim_model)
        res_weight_init_scale = (2 * blocks)**(-0.5)
        self.decoder_blocks = [TransformerBlock(dim_model, n_heads, ln_position, res_weight_init_scale) for _ in range(blocks)]

        # Final Layer
        self.final_layernorm = LayerNorm(dim_model)
        self.final_linear = Linear(dim_model, self.vocab_size)
        self.loss_function = CrossEntropyWithSoftmax()

    def forward(self, input_tokens_decoder, mask_attention = None, padding_mask = None):
        # Decoder forward
        out_d = self.embedding_decoder.forward(input_tokens_decoder)
        out_d = self.positional_encodings_decoder.forward(out_d)

        for decoder in self.decoder_blocks:
            out_d = decoder.forward(
                inputs=out_d,
                mask_attention=mask_attention,
                padding_mask=padding_mask
            )

        out = self.final_layernorm.forward(out_d)
        return self.final_linear.forward(out)
