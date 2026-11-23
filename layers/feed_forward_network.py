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
