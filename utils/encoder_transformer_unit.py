import numpy as np
import tensorflow as tf
from typing import Callable

from .multi_head_attention import MultiHeadAttention


class EncoderTransformerUnit(tf.keras.layers.Layer):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_inner: int = 2048,
        activation_inner: Callable = tf.nn.relu,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # One encoder unit consists of:

        # Sub layer 1:
        # - MultiHeadAttention
        # - Dropout
        # - Residual (add to input)
        # - Normalize
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads,
            dim_model=dim_model,
        )
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.norm_1 = tf.keras.layers.BatchNormalization()

        # Sub layer 2:
        # - position wise feed forward (2 layers with ReLu in middle)
        # - Dropout
        # - Residual (add to input)
        # - Normalize
        self.position_wise_feed_forward_inner = tf.keras.layers.Dense(dim_inner, activation=activation_inner)
        self.position_wise_feed_forward_outer = tf.keras.layers.Dense(dim_model)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.norm_2 = tf.keras.layers.BatchNormalization()

    def call(self, x, mask=None):
        # Multi-head attention.
        multi_head_attn_output = self.multi_head_attention(x, mask=mask)
        # Dropout.
        multi_head_attn_output = self.dropout_1(multi_head_attn_output)
        # Residual.
        multi_head_attn_output = multi_head_attn_output + x
        # Batch norm.
        multi_head_attn_output = self.norm_1(multi_head_attn_output)

        # Feed forward 1 (w/ relu).
        outputs = self.position_wise_feed_forward_inner(multi_head_attn_output)
        # Feed forward 2 (w/o relu).
        outputs = self.position_wise_feed_forward_outer(outputs)
        # Dropout.
        outputs = self.dropout_2(outputs)
        # Residual.
        outputs = outputs + multi_head_attn_output
        # Batch norm.
        outputs = self.norm_2(outputs)

        return outputs


if __name__ == "__main__":
    B = 2
    T = 3
    d_model = 8

    encoder_unit = EncoderTransformerUnit(dim_model=d_model)

    np.random.seed(42)
    embedding_out = np.random.sample(size=(B, T, d_model))

    out = encoder_unit(embedding_out)
    print(out)
