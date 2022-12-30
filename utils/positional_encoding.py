import math
import matplotlib.pyplot as plt
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, dim_model, max_seq_len=5000, **kwargs):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__(**kwargs)

        # Create matrix of [T, dim_model] representing the positional encoding
        # for max_seq_len inputs.
        #pe = tf.zeros([max_seq_len, dim_model], tf.float32)
        #position = tf.expand_dims(tf.range(0, max_seq_len, dtype=tf.float32), axis=1)

        # Compute the division term to be sin'd/cos'd and then added to the
        # embedding output.
        div_terms = tf.math.exp(
            tf.range(0, dim_model, 2, dtype=tf.float32) * (-math.log(10000.0) / dim_model)
        )

        # Set all even positions to sin(..).
        # Set all odd positions to cos(..).
        positional_encoding = []
        for pos in range(max_seq_len):
            for i in range(dim_model):
                positional_encoding.append(
                    math.sin(pos * div_terms[i // 2]) if i % 2 == 0
                    else math.cos(pos * div_terms[i // 2])
                )
        positional_encoding = tf.stack(positional_encoding)
        positional_encoding = tf.reshape(positional_encoding, [max_seq_len, dim_model])

        # [B = 1, T, d_model]
        self.positional_encoding = tf.expand_dims(positional_encoding, axis=0)

    def forward(self, x):
        # x=[B, T, d_model]
        # pos_encoding=[B, T, d_model]
        x = x + self.positional_encoding[:, :x.shape[1]]
        return x


if __name__ == "__main__":
    encod_block = PositionalEncoding(dim_model=48, max_seq_len=96)
    pe = tf.transpose(encod_block.positional_encoding[0]).numpy()  # [0] b/c B=1

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    pos = ax.imshow(pe, cmap="RdGy", extent=(1, pe.shape[1] + 1, pe.shape[0] + 1, 1))
    fig.colorbar(pos, ax=ax)

    ax.set_title("Positional encoding over hidden dimensions")

    ax.set_xlabel("Position in sequence")
    ax.set_xticks([1] + [i * 10 for i in range(1, 1 + pe.shape[1] // 10)])

    ax.set_ylabel("Hidden dimension")
    ax.set_yticks([1] + [i * 10 for i in range(1, 1 + pe.shape[0] // 10)])

    plt.show()

    print("")
