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
