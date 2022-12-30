import tensorflow as tf
from typing import Optional

from utils.decoder_block import DecoderBlock
from utils.encoder_block import EncoderBlock
from utils.positional_encoding import PositionalEncoding


class AttentionIsAllYouNeedModel(tf.keras.models.Model):
    """Implementation of the model described in https://arxiv.org/pdf/1706.03762.pdf

    Attention is all you need. Vaswani et al. 2017
    """
    def __init__(
        self,
        input_dim: int = 1000,
        use_embedding: bool = True,
        max_seq_len: int = 64,
        num_encoder_units: int = 6,
        num_decoder_units: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_inner_ffn: int = 2048,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_heads = num_heads

        #TEST
        self.input_batchnorm = tf.keras.layers.BatchNormalization()
        #END TEST

        # Embeddings are shared between input and output.
        # Also, the final linear uses the same learned weights as are in the embedding
        # (transposed).
        self.in_and_out_embedding = None
        if use_embedding:
            self.in_and_out_embedding = tf.keras.layers.Embedding(
                input_dim=input_dim,
                output_dim=dim_model,
            )
        elif input_dim != dim_model:
            self.in_and_out_embedding = tf.keras.layers.Dense(
                dim_model,
                activation=None,
                use_bias=False,
            )

        self.positional_encoding = PositionalEncoding(
            dim_model=dim_model,
            max_seq_len=max_seq_len,
        )

        self.encoder_block = EncoderBlock(
            num_units=num_encoder_units,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_inner=dim_inner_ffn,
            dropout_rate=dropout_rate,
        )

        self.decoder_block = DecoderBlock(
            num_units=num_decoder_units,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_inner=dim_inner_ffn,
            dropout_rate=dropout_rate,
        )

        self.self_attention_mask = tf.Variable(
            initial_value=[
                [1.0 if x > y else 0.0 for x in range(max_seq_len)]
                for y in range(max_seq_len)
            ],
            dtype = tf.float32,
            trainable=False,
        )

    def call(self, inputs, outputs, seq_mask):
        #TEST
        inputs = self.input_batchnorm(inputs)
        #END TEST

        # Compute embedding output.
        if self.in_and_out_embedding is not None:
            inputs = self.in_and_out_embedding(inputs)
        inputs = self.positional_encoding(inputs)

        # Fix `seq_mask` so it matches [B, h, T, T] structure of our logits.
        encoder_mask = tf.einsum("bt,bu->btu", seq_mask, seq_mask)
        # mask=[B, T, T]
        # Add head dim, after B.
        encoder_mask = tf.tile(tf.expand_dims(encoder_mask, 1), multiples=[1, self.num_heads, 1, 1])
        # mask=[B, h, T, T]

        # Compute encoder output.
        encoder_out = self.encoder_block(inputs, mask=encoder_mask)

        # Compute decoder output.
        # Mix [B, T] seq_mask with [T, T] self_attention_mask
        decoder_mask = tf.einsum("bhtu,tu->bhut", tf.cast(encoder_mask, tf.float32), self.self_attention_mask)

        if self.in_and_out_embedding is not None:
            outputs = self.in_and_out_embedding(outputs)
        outputs = self.positional_encoding(outputs)

        decoder_out = self.decoder_block(outputs, encoder_out, mask=decoder_mask)

        # Push through final linear (which shares its weights with both embedding
        # layers for in- and outputs).
        if self.in_and_out_embedding is not None:
            if hasattr(self.in_and_out_embedding, "embeddings"):
                decoder_out = tf.matmul(decoder_out, self.in_and_out_embedding.embeddings, transpose_b=True)
            else:
                decoder_out = tf.matmul(decoder_out, self.in_and_out_embedding.kernel, transpose_b=True)

        # Return output probabilities (to sample from).
        return tf.nn.softmax(decoder_out)


if __name__ == "__main__":
    attention_net = AttentionIsAllYouNeedModel()
