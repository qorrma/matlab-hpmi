from abc import ABC
import argparse
import yaml

from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.layers import Conv2D, ReLU, UpSampling2D, Dense
from data_manage import batch_dataset
from callbacks import SaveOutputImagesCallback, SaveModelCallback
from tensorflow.keras.experimental import CosineDecayRestarts
from tensorflow.keras.utils import plot_model

import tensorflow as tf
from tensorflow import keras


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
AUTO = tf.data.experimental.AUTOTUNE


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, lamda=0.5):
        super().__init__()
        self.lamda = lamda

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(self.lamda * z_log_var) * epsilon


def ConvVAE(
        input_shape=(256, 256, 1),
        num_layer=4,
        init_embed=8,
        extension=1
):
    in_channel = input_shape[2]
    channels = [(2**i)*init_embed*extension for i in range(num_layer)]
    bias = True

    latent_dim = input_shape[0] // (2 ** num_layer)
    # Encoder
    inputs = Input(shape=input_shape)
    x = inputs
    for i in range(num_layer):
        x = Conv2D(
            filters=channels[i],
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=bias)(x)
        x = ReLU()(x)

    x = tf.keras.backend.reshape(x, shape=(-1, latent_dim * latent_dim * channels[-1]))
    x = Dense(latent_dim * latent_dim * channels[-1], activation='relu')(x)
    # x = keras.layers.LayerNormalization()(x)

    z_mean = Dense(channels[-1], name='fc1')(x)
    z_log_val = Dense(channels[-1], name='fc2')(x)
    x = Sampling()([z_mean, z_log_val])

    x = Dense(latent_dim * latent_dim * channels[-1], activation='relu')(x)
    x = tf.keras.backend.reshape(x, shape=(-1, latent_dim, latent_dim, channels[-1]))
    # Decoder
    for i in range(num_layer-1, 0, -1):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(
            filters=channels[i],
            kernel_size=(3, 3),
            padding='same',
            use_bias=bias)(x)
        x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(
        filters=in_channel,
        kernel_size=(3, 3),
        padding='same',
        use_bias=bias)(x)
    outputs = tf.keras.backend.sigmoid(x)
    # Create the model
    model = Model(inputs=inputs, outputs=[outputs, z_mean, z_log_val], name='vae')
    # model.summary()
    plot_model(model, to_file='Variantional_autoencoder_architecture.png', show_shapes=True)
    return model


class VAE(keras.Model, ABC):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.train_meter = keras.metrics.Mean(name="total_loss")
        self.mse_meter = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_meter = keras.metrics.Mean(name="kl_loss")
        self.test_meter = keras.metrics.Mean(name="reconstruction_loss")

    @property
    def metrics(self):
        return [self.train_meter, self.test_meter, self.mse_meter, self.kl_meter]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction, z_mean, z_log_var = self.model(data)
            mse_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = mse_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.train_meter.update_state(total_loss)
        self.mse_meter.update_state(mse_loss)
        self.kl_meter.update_state(kl_loss)

        return {
            "kl_loss": self.kl_meter.result(),
            "rec_loss": self.mse_meter.result(),
            "loss": self.train_meter.result()}

    def test_step(self, data):
        reconstruction, z_mean, z_log_var = self.model(data)
        mse_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2)
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = mse_loss + kl_loss
        self.test_meter.update_state(total_loss)
        return {"loss": self.test_meter.result()}


def train(
        root='../data',
        dataset='LXSemicon',
        bgl='48'
):
    with open('../configs/VAE.yaml') as f:
        cfg = yaml.full_load(f)

    batch_size = cfg["DATA"]["batch_size"]
    seed = cfg["DATA"]["seed"]

    epochs = cfg["OPTIM"]["epochs"]
    lr = cfg["OPTIM"]["learning_rate"]

    input_shape = cfg["MODEL"]["input_shape"]
    extension = cfg["MODEL"]["extension"]
    init_embed = cfg["MODEL"]["init_embed"]
    n_layers = cfg["MODEL"]["n_layers"]

    tf.random.set_seed(seed)
    train_ds = batch_dataset(f'{root}/{dataset}/{bgl}/', batch_size)
    test_ds = batch_dataset(f'{root}/private/{bgl}/', batch_size, False)

    model = ConvVAE(
        input_shape=input_shape,
        num_layer=n_layers,
        init_embed=init_embed,
        extension=extension)
    cae = VAE(model)

    decay_epochs = 50
    cos_decay_ann = CosineDecayRestarts(
        initial_learning_rate=lr,
        first_decay_steps=len(train_ds) * decay_epochs,
        t_mul=1, m_mul=0.5, alpha=0.0001
    )

    cae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cos_decay_ann)
    )

    logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    save_cb = SaveOutputImagesCallback(
        t_model=model,
        alg='vae',
        save_dir=f'VAE_checkpoints/{dataset}_{bgl}',
        save_frequency=10,
        train_ds=train_ds,
        test_ds=test_ds
    )

    ckpt_cb = SaveModelCallback(
        t_model=model,
        save_dir='VAE_checkpoints/',
        save_name=f'{dataset}_{bgl}_best.h5',
        save_frequency=10,
    )

    cae.fit(
        train_ds,
        validation_data=test_ds,
        validation_freq=10,
        epochs=epochs,
        callbacks=[
            ckpt_cb,
            save_cb,
            tensorboard_cb,
        ]
    )


def ArgumentsParse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bgl", default='235', type=str, help='background gray level')
    return parser.parse_args()


if __name__ == '__main__':
    args = ArgumentsParse()
    train(bgl=args.bgl)
