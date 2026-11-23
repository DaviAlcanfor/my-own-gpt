from layers import GPTLayer, EmbeddingLayer, Softmax, CrossEntropyWithSoftmax

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
