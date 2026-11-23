class ReLU:
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, prev_grad):
        return np.maximum(0, np.sign(self.inputs)) * prev_grad
