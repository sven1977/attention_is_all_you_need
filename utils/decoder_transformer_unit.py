import numpy as np
import tensorflow as tf

from .multi_head_attention import MultiHeadAttention


class DecoderTransformerUnit(tf.keras.layers.Layer):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_inner: int = 2048,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # One decoder unit consists of:

        # Sub layer 1:
        # - MultiHeadAttention
        # - Dropout
        # - Residual (add to input)
        # - Normalize
        self.multi_head_attention_1 = MultiHeadAttention(
            num_heads=num_heads,
            dim_model=dim_model,
        )
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.norm_1 = tf.keras.layers.BatchNormalization()

        # Sub layer 2:
        # - Masked MultiHeadAttention
        # - Dropout
        # - Residual (add to input)
        # - Normalize
        self.multi_head_attention_2 = MultiHeadAttention(
            num_heads=num_heads,
            dim_model=dim_model,
            # For the middle multi-head layer, keys and values are generated from
            # encoder output data. Queries are computed from previous decoder layer.
            k_and_v_from_encoder=True,
        )
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.norm_2 = tf.keras.layers.BatchNormalization()

        # Sub layer 3:
        # - position wise feed forward (2 layers with ReLu in middle)
        # - Dropout
        # - Residual (add to input)
        # - Normalize
        self.position_wise_feed_forward_inner = tf.keras.layers.Dense(dim_inner, activation=tf.nn.relu)
        self.position_wise_feed_forward_outer = tf.keras.layers.Dense(dim_model)
        self.dropout_3 = tf.keras.layers.Dropout(dropout_rate)
        self.norm_3 = tf.keras.layers.BatchNormalization()

    def call(self, x, encoder_output, mask):
        # Masked multi-head attention.
        multi_head_attn_output_1 = self.multi_head_attention_1(x, mask=mask)
        # Dropout.
        multi_head_attn_output_1 = self.dropout_1(multi_head_attn_output_1)
        # Residual.
        multi_head_attn_output_1 = multi_head_attn_output_1 + x
        # Batch norm.
        multi_head_attn_output_1 = self.norm_1(multi_head_attn_output_1)

        # Multi-head attention.
        multi_head_attn_output_2 = self.multi_head_attention_2(multi_head_attn_output_1, encoder_output=encoder_output)
        # Dropout.
        multi_head_attn_output_2 = self.dropout_2(multi_head_attn_output_2)
        # Residual.
        multi_head_attn_output_2 = multi_head_attn_output_2 + multi_head_attn_output_1
        # Batch norm.
        multi_head_attn_output_2 = self.norm_2(multi_head_attn_output_2)

        # Feed forward 1 (has relu).
        outputs = self.position_wise_feed_forward_inner(multi_head_attn_output_2)
        # Feed forward 2 (no relu).
        outputs = self.position_wise_feed_forward_outer(outputs)
        # Dropout.
        outputs = self.dropout_3(outputs)
        # Residual.
        outputs = outputs + multi_head_attn_output_2
        # Batch norm.
        outputs = self.norm_3(outputs)

        return outputs


if __name__ == "__main__":
    B = 2
    T = 3
    d_model = 8

    decoder_unit = DecoderTransformerUnit(dim_model=d_model)

    np.random.seed(42)
    embedding_out = np.random.sample(size=(B, T, d_model))
    encoder_output = np.random.sample(size=(B, T, d_model))

    out = decoder_unit(embedding_out, encoder_output=encoder_output)
    print(out)
