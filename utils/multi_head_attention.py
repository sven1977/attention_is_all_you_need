import numpy as np
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, num_heads: int = 8, dim_model: int = 512, k_and_v_from_encoder: bool = False, **kwargs):
        super().__init__(**kwargs)

        # Number of attention heads to use.
        self.num_heads = num_heads
        # Dimensionality of the model (input and output dim).
        self.dim_model = dim_model

        # Whether this is an "encoder-decoder attention" layer. In this case,
        # a) this layer sits inside a decoder block,
        # b) keys and values are generated from the previous decoder
        # multi-head-attention layer, and c) queries are generated from the encoder
        # output.
        self.k_and_v_from_encoder = k_and_v_from_encoder

        # Dimensionality of the linearly projected queries, keys, and values.
        # Note: Keys, queries, and values have all the same dimension in our
        # implementation
        assert self.dim_model % self.num_heads == 0, (
            "`dim_model` must be 0 modulo `num_heads`."
        )
        self.dim_head = self.dim_model // self.num_heads

        # Note that in many implementations you see "bias=False" for the following
        # two dense layers which is optional.

        # Stack all weight matrices (for all heads 1...h) together for efficiency.
        # Output of the qkv layer will be [B, T, h, 3*d_model], where B, T, h are
        # all treated
        if k_and_v_from_encoder:
            self.qkv_proj = None
            self.kv_proj = tf.keras.layers.Dense(2 * self.num_heads * self.dim_head)
            self.q_proj = tf.keras.layers.Dense(1 * self.num_heads * self.dim_head)
        else:
            self.qkv_proj = tf.keras.layers.Dense(3 * self.num_heads * self.dim_head)
            self.kv_proj = self.q_proj = None
        # Single linear layer for mapping concatenated (over heads) output of
        # multi-head attention back to [B, T, dim_model].
        self.o_proj = tf.keras.layers.Dense(self.dim_model)

    @staticmethod
    def _scaled_dot_product(q, k, v, mask=None):
        d_head = q.shape[-1]
        num_heads = q.shape[1]
        # q=[B, h, T, d_head]
        # k=[B, h, T, d_head] -> transpose to [B, h, d_head, T]
        # q x k=[B, h, T, T] (self-attention: each token attends to all tokens).
        attn_logits = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))
        # Scale by square root of d_head.
        # attn_logits=[B, h, T, T].
        attn_logits = attn_logits / tf.math.sqrt(tf.cast(d_head, tf.float32))

        # Mask, if necessary.
        if mask is not None:
            # mask=[B, h, T, T]
            attn_logits = tf.where(tf.cast(mask, tf.bool), attn_logits, tf.ones_like(attn_logits) * -9e15)

        # Softmax to get weights (for the values) that sum to 1.0.
        attention = tf.nn.softmax(attn_logits, axis=-1)
        # Compute weighted values.
        # attention=[B, h, T, T].
        # v=[B, h, T, d_head]
        # weighted_values=[B, h, T, d_head] (matmul 2 inner dimensions)
        weighted_values = tf.matmul(attention, v)

        return weighted_values, attention

    def call(self, x, encoder_output=None, mask=None, return_attention=False):
        B, T, dim = x.shape
        assert dim == self.dim_model, f"Input shape of `x` ({x.shape}) incorrect!"
        if encoder_output is not None:
            assert encoder_output.shape == x.shape
            assert self.k_and_v_from_encoder is True

        # All inputs (q, k, v) from from previous (encoder or decoder) layer.
        if encoder_output is None:
            qkv = self.qkv_proj(x)
            # For each batch item, seq position, and head: Compute a Q, K, and V vector
            # of dim=self.dim_head.
            qkv = tf.reshape(qkv, [B, T, self.num_heads, 3 * self.dim_head])
            # Transpose for correct dot product between keys and queries:
            # Dot product's result should be [B, T, T, dim_head].
            qkv = tf.transpose(qkv, perm=[0, 2, 1, 3])  # [B, h, T, dim_model]
            # Separate Q, K, V from linear output.
            # Each q, k, and v now is [B, h, T, dim_head].
            q, k, v = tf.split(qkv, 3, axis=-1)
        # q from prev decoder layer, k and v from encoder output.
        # Completely analogous to above case.
        else:
            q = self.q_proj(x)
            q = tf.reshape(q, [B, T, self.num_heads, self.dim_head])
            q = tf.transpose(q, perm=[0, 2, 1, 3])  # [B, h, T, dim_model]

            kv = self.kv_proj(encoder_output)
            kv = tf.reshape(kv, [B, T, self.num_heads, 2* self.dim_head])
            kv = tf.transpose(kv, perm=[0, 2, 1, 3])  # [B, h, T, dim_model]

            k, v = tf.split(kv, 2, axis=-1)

        # Determine value outputs
        values, attention = self._scaled_dot_product(q, k, v, mask=mask)
        # Transform back to [B, T, h, dim_head].
        values = tf.transpose(values, perm=[0, 2, 1, 3])
        # Concatenation: Same as reshaping back to [B, T, dim_model].
        values = tf.reshape(values, [B, T, self.dim_model])

        # Map through last dense layer.
        output = self.o_proj(values)

        if return_attention:
            return output, attention
        else:
            return output


if __name__ == "__main__":
    multi_head_attention = MultiHeadAttention(dim_model=8)

    B = 2
    T = 3
    d_embed = multi_head_attention.dim_model

    np.random.seed(42)
    embedding_out = np.random.sample(size=(B, T, d_embed))

    out = multi_head_attention(embedding_out)
    print(out)
