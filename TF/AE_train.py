from abc import ABC
import argparse
import yaml

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.layers import Reshape, Dense, Flatten
from TFsource.data_manage import batch_dataset
from TFsource.callbacks import SaveOutputImagesCallback, SaveModelCallback
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


def AE(
        input_shape=(256, 256, 1),
        extension=1
):
    output_dim = input_shape[0] * input_shape[1] * input_shape[2]
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(1024 * extension, activation='relu')(x)
    x = Dense(512 * extension, activation='relu')(x)
    x = Dense(256 * extension, activation='relu')(x)
    encode = Dense(128, activation='relu')(x)

    # z_mean = Dense(128, name='fc1')(x)
    # z_log_val = Dense(128, name='fc2')(x)
    # x = Sampling()([z_mean, z_log_val])

    x = Dense(256 * extension, activation='relu')(encode)
    x = Dense(512 * extension, activation='relu')(x)
    x = Dense(1024 * extension, activation='relu')(x)
    x = Dense(output_dim, activation='sigmoid')(x)
    decode = Reshape(target_shape=input_shape)(x)

    # Create the model
    model = Model(inputs=inputs, outputs=[decode], name='ae')
    model.summary()
    # plot_model(model, to_file='Variantional_autoencoder_architecture.png', show_shapes=True)
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
        lr = self.optimizer.lr(self.optimizer.iterations)
        with tf.GradientTape() as tape:
            reconstruction = self.model(data)
            loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2)
                )
            )
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.train_meter.update_state(loss)
        return {
            "lr": lr,
            "loss": self.train_meter.result()}

    def test_step(self, data):
        reconstruction = self.model(data)
        loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2)
            )
        )
        self.test_meter.update_state(loss)
        return {"loss": self.test_meter.result()}


def train(
        root='../data',
        dataset='LXSemicon',
        bgl='48'
):
    with open('../configs/AE.yaml') as f:
        cfg = yaml.full_load(f)

    batch_size = cfg["DATA"]["batch_size"]
    seed = cfg["DATA"]["seed"]

    epochs = cfg["OPTIM"]["epochs"]
    lr = cfg["OPTIM"]["learning_rate"]

    input_shape = cfg["MODEL"]["input_shape"]
    extension = cfg["MODEL"]["extension"]

    tf.random.set_seed(seed)
    train_ds = batch_dataset(f'{root}/private/{bgl}/', batch_size)
    test_ds = batch_dataset(f'{root}/private/{bgl}/', batch_size, False)

    model = AE(input_shape=input_shape, extension=extension)
    cae = VAE(model)

    decay_epochs = 20
    cos_decay_ann = CosineDecayRestarts(
        initial_learning_rate=lr,
        first_decay_steps=len(train_ds) * decay_epochs,
        t_mul=1, m_mul=0.9, alpha=0
    )

    cae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cos_decay_ann),
    )

    save_cb = SaveOutputImagesCallback(
        t_model=model,
        alg='ae',
        save_dir=f'AE_checkpoints/{dataset}_{bgl}',
        save_frequency=10,
        train_ds=train_ds,
        test_ds=test_ds
    )

    ckpt_cb = SaveModelCallback(
        t_model=model,
        save_dir='AE_checkpoints/',
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
        ]
    )


def ArgumentsParse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bgl", default='218', type=str, help='background gray level')
    return parser.parse_args()


if __name__ == '__main__':
    args = ArgumentsParse()
    train(bgl=args.bgl)
