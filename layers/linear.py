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
