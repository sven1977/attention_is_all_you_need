import tensorflow as tf

from .decoder_transformer_unit import DecoderTransformerUnit


class DecoderBlock(tf.keras.layers.Layer):
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

        self.decoder_units = [
            DecoderTransformerUnit(
                dim_model=dim_model,
                num_heads=num_heads,
                dim_inner=dim_inner,
                dropout_rate=dropout_rate,
            ) for _ in range(num_units)
        ]

    def call(self, x, encoder_output, mask, **kwargs):
        for unit in self.decoder_units:
            x = unit(x, encoder_output=encoder_output, mask=mask)
        return x

    def get_attention_maps(self, x, encoder_output, mask):
        """Convenience method to get the [B, h, TxT] attention (softmax'd weights)."""
        attention_maps = []
        for unit in self.decoder_units:
            _, attn_map = unit.multi_head_attention(
                x,
                encoder_output=encoder_output,
                mask=mask,
                return_attention=True,
            )
            attention_maps.append(attn_map)
            x = unit(
                x,
                encoder_output=encoder_output,
                mask=mask,
            )
        return attention_maps
