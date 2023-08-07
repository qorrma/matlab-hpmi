from abc import ABC
import argparse
import yaml
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, ReLU, UpSampling2D, Input, Dense
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


def ConvAE(input_shape=(256, 256, 1), num_layer=4, init_embed=8, extension=1):
    in_channel = input_shape[2]
    channels = [(2**i)*init_embed*extension for i in range(num_layer)]
    bias = True

    latent_dim = input_shape[0] // (2 ** num_layer)

    def gaussian_kernel(size=3, sigma=1):
        x_range = tf.range(-(size - 1) // 2, (size - 1) // 2 + 1, 1)
        y_range = tf.range((size - 1) // 2, -(size - 1) // 2 - 1, -1)

        xs, ys = tf.meshgrid(x_range, y_range)
        kernel = tf.exp(-(xs ** 2 + ys ** 2) / (2 * (sigma ** 2))) / (
                    2 * np.pi * (sigma ** 2))
        return tf.cast(kernel / tf.reduce_sum(kernel), tf.float32)

    kernel = gaussian_kernel(11, 1)
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)

    # Encoder
    inputs = Input(shape=input_shape)
    x = inputs
    x = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
    for i in range(num_layer):
        x = Conv2D(
            filters=channels[i],
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=bias)(x)
        x = ReLU()(x)

    x = tf.keras.backend.reshape(
        x, shape=(-1, latent_dim * latent_dim * channels[-1]))
    x = Dense(latent_dim * latent_dim * channels[-1], activation='relu')(x)
    x = keras.layers.LayerNormalization()(x)
    x = tf.keras.backend.reshape(
        x, shape=(-1, latent_dim, latent_dim, channels[-1]))

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
    decode = tf.keras.backend.sigmoid(x)
    # Create the model
    model = Model(inputs=inputs, outputs=decode, name='cae')
    # model.summary()
    plot_model(model, to_file='Convolutional_autoencoder_architecture.png', show_shapes=True)
    return model


class CAE(keras.Model, ABC):
    """
    ## Define the VAE as a `Model` with a custom `train_step`
    """

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.train_meter = keras.metrics.Mean(name="reconstruction_loss")
        self.test_meter = keras.metrics.Mean(name="reconstruction_loss")

    @property
    def metrics(self):
        return [self.train_meter]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction = self.model(data)

            err = tf.squeeze(data - reconstruction)
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(err), axis=(1, 2)))

            # loss = tf.reduce_mean(
            #     tf.reduce_sum(
            #         keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2)
            #     )
            # )
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.train_meter.update_state(loss)
        return {"loss": self.train_meter.result()}

    def test_step(self, data):
        reconstruction = self.model(data)
        err = tf.squeeze(data - reconstruction)
        loss = tf.reduce_mean(tf.reduce_mean(tf.square(err), axis=(1, 2)))

        self.test_meter.update_state(loss)
        return {"loss": self.test_meter.result()}


def train(
        root=r'D:\A.SCI\[1st] Measurment\code\data',
        dataset='LXSemicon',
        bgl='48'
):
    with open('../configs/CAE.yaml') as f:
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

    model = ConvAE(
        input_shape=input_shape,
        num_layer=n_layers,
        init_embed=init_embed,
        extension=extension)
    cae = CAE(model)

    decay_epochs = 50
    cos_decay_ann = CosineDecayRestarts(
        initial_learning_rate=lr,
        first_decay_steps=len(train_ds) * decay_epochs,
        t_mul=1, m_mul=0.5, alpha=0.0001
    )

    cae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cos_decay_ann)
    )

    save_cb = SaveOutputImagesCallback(
        t_model=model,
        alg='cae',
        save_dir=f'CAE_checkpoints/{dataset}_{bgl}',
        save_frequency=10,
        train_ds=train_ds,
        test_ds=test_ds
    )
    ckpt_cb = SaveModelCallback(
        t_model=model,
        save_dir='CAE_checkpoints/',
        save_name=f'{dataset}_{bgl}_best.h5',
        save_frequency=10,
    )
    stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        verbose=1,
        min_delta=0.1,
        patience=30,
    )
    # best_cb = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f'CAE_checkpoints/{dataset}_{bgl}/',
    #     save_freq=10,
    #     monitor='val_loss',
    #     mode='min',
    # )
    cae.fit(
        train_ds,
        validation_data=test_ds,
        validation_freq=10,
        shuffle=False,
        epochs=epochs,
        callbacks=[
            ckpt_cb,
            save_cb,
            stop_cb,
            # best_cb,
        ]
    )


def ArgumentsParse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bgl", default='128', type=str, help='background gray level')
    return parser.parse_args()


if __name__ == '__main__':
    args = ArgumentsParse()
    train(bgl=args.bgl)
