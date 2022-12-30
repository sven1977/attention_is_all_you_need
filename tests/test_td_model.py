import gymnasium as gym
from keras.utils.np_utils import to_categorical
import numpy as np
import tensorflow as tf

from attention_is_all_you_need_model import AttentionIsAllYouNeedModel
from utils.env_runner import EnvRunner
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig


class MyLrSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __call__(self, step):
        if step < 10:
            return (((0.0001 - 0.0) / 10) * step) + 0.0
        elif step < 110000:
            return (((0.00001 - 0.0001) / (110000-10)) * (step - 10)) + 0.0001
        else:
            return 0.00001


class CountingCartPole(gym.ObservationWrapper):
    def __init__(self):
        env = gym.make("CartPole-v0")
        super().__init__(env=env)
        self._ts = 0

    def reset(self, **kwargs):
        self._ts = 0
        return super().reset(**kwargs)

    def step(self, action):
        self._ts += 1
        return super().step(action)

    def observation(self, observation):
        return np.array(list(observation) + [self._ts])

    @property
    def observation_space(self):
        low = np.array(list(self.env.observation_space.low) + [0.0])
        high = np.array(list(self.env.observation_space.high) + [1000.0])
        return gym.spaces.Box(low, high, (5,), dtype=np.float32)


gym.register("CountingCartPole-v0", CountingCartPole)

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=MyLrSchedule())

max_seq_len = 150

config = (
    AlgorithmConfig()
    .environment("CountingCartPole-v0")
    .rollouts(rollout_fragment_length=1000, num_envs_per_worker=2)
)
env_runner = EnvRunner(config=config, max_seq_len=max_seq_len)

model = AttentionIsAllYouNeedModel(
    use_embedding=False,  # CartPole doesn't need embedding
    input_dim=env_runner.env.single_observation_space.shape[0] + env_runner.env.single_action_space.n,
    max_seq_len=max_seq_len - 1,  # -1 b/c we need to extract next obs from data
    num_encoder_units=2,
    num_decoder_units=2,
    dim_model=32,
    num_heads=2,
    dim_inner_ffn=256,
    dropout_rate=0.0,
)
num_iterations = 100

for iteration in range(num_iterations):
    # Gather some data from the environment.
    observation_batch, action_batch, reward_batch, masks = env_runner.sample()
    # Concatenate observations and actions per timestep.
    input_batch = np.concatenate(
        [
            observation_batch[:, :-1],
            to_categorical(action_batch[:, :-1], env_runner.env.single_action_space.n)
        ],
        axis=2,
    )
    # Concatenate next observations and next actions per timestep.
    output_batch = np.concatenate(
        [
            observation_batch[:, 1:],
            to_categorical(action_batch[:, 1:], env_runner.env.single_action_space.n)
        ],
        axis=2,
    )
    with tf.GradientTape() as tape:
        predictions = model(
            inputs=input_batch,
            outputs=output_batch,
            seq_mask=masks[:, :-1],
        )
        if iteration == 0:
            model.summary()
        # Ignore outputs beyond actual sequences for loss computations.
        masked_predictions = tf.einsum("btd,bt->btd", predictions, masks[:, 1:])
        loss = loss_fn(masked_predictions[:, :, :4], observation_batch[:, 1:, :4])
        print(f"{iteration}) B={input_batch.shape[0]} L={loss} lr={optimizer.learning_rate(iteration)}")
        dL_dVars = tape.gradient(loss, model.trainable_variables)
        assert len(dL_dVars) == len(model.trainable_variables)
        optimizer.apply_gradients(
            [(g, v) for g, v in zip(dL_dVars, model.trainable_variables)]
        )
