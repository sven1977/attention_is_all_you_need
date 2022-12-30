import tensorflow as tf

from .encoder_transformer_unit import EncoderTransformerUnit


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            num_units: int = 6,
            dim_model: int = 512,
            num_heads: int = 8,
            dim_inner: int = 2048,
            dropout_rate: float = 0.1,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.encoder_units = [
            EncoderTransformerUnit(
                dim_model=dim_model,
                num_heads=num_heads,
                dim_inner=dim_inner,
                dropout_rate=dropout_rate,
            ) for _ in range(num_units)
        ]

    def call(self, x, mask=None, **kwargs):
        for unit in self.encoder_units:
            x = unit(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        """Convenience method to get the [B, h, TxT] attention (softmax'd weights)."""
        attention_maps = []
        for unit in self.encoder_units:
            _, attn_map = unit.multi_head_attention(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = unit(x, mask=mask)
        return attention_maps
