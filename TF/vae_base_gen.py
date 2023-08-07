from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import copy
import yaml
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.layers import Conv2D, ReLU, UpSampling2D, Conv2DTranspose, Dense
from tensorflow.keras.utils import plot_model
from PIL import Image, ImageFilter
import tensorflow as tf


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

    # def gaussian_kernel(size=3, sigma=1):
    #     x_range = tf.range(-(size - 1) // 2, (size - 1) // 2 + 1, 1)
    #     y_range = tf.range((size - 1) // 2, -(size - 1) // 2 - 1, -1)
    #
    #     xs, ys = tf.meshgrid(x_range, y_range)
    #     kernel = tf.exp(-(xs ** 2 + ys ** 2) / (2 * (sigma ** 2))) / (
    #                 2 * np.pi * (sigma ** 2))
    #     return tf.cast(kernel / tf.reduce_sum(kernel), tf.float32)
    #
    # kernel = gaussian_kernel(11, 1)
    # kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)

    # Encoder
    inputs = Input(shape=input_shape)
    x = inputs
    # x = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
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
        x = Conv2DTranspose(
            filters=channels[i],
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='same',
            use_bias=bias)(x)
        x = ReLU()(x)

    x = Conv2DTranspose(
        filters=in_channel,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding='same',
        use_bias=bias)(x)

    # for i in range(num_layer-1, 0, -1):
    #     x = UpSampling2D(size=(2, 2))(x)
    #     x = Conv2D(
    #         filters=channels[i],
    #         kernel_size=(3, 3),
    #         padding='same',
    #         use_bias=bias)(x)
    #     x = ReLU()(x)
    #
    # x = UpSampling2D(size=(2, 2))(x)
    # x = Conv2D(
    #     filters=in_channel,
    #     kernel_size=(3, 3),
    #     padding='same',
    #     use_bias=bias)(x)
    outputs = tf.keras.backend.sigmoid(x)
    # Create the model
    model = Model(inputs=inputs, outputs=[outputs, z_mean, z_log_val], name='vae')
    # model.summary()
    plot_model(model, to_file='Variantional_autoencoder_architecture.png', show_shapes=True)
    return model


def fit_transform(X: np.ndarray, bgl=48):
    with open('../configs/VAE.yaml') as f:
        cfg = yaml.full_load(f)

    input_shape = cfg["MODEL"]["input_shape"]
    extension = cfg["MODEL"]["extension"]
    init_embed = cfg["MODEL"]["init_embed"]
    n_layers = cfg["MODEL"]["n_layers"]

    model = ConvVAE(
        input_shape=input_shape,
        num_layer=n_layers,
        init_embed=init_embed,
        extension=extension)
    model.load_weights(f'./VAE_checkpoints/LXSemicon_{bgl}_best.h5')

    x = copy.deepcopy(X)
    x = np.expand_dims(x, 0)
    x = np.expand_dims(x, -1).astype(np.float)/255.0

    base, _, _ = model.predict(x)
    base = base[0, :, :, 0] * 255.0
    cmap = np.abs(X - base) / base
    return base, cmap


def save_img(filepath, data, minmax=False):
    if minmax:
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255.0
        img = Image.fromarray(data.astype(np.uint8))
        img.save(filepath)
    else:
        img = Image.fromarray(data.astype(np.uint8))
        img.save(filepath)


def load_img(filepath):
    # Load images by using PIL image format
    return Image.open(filepath).convert('L')


if __name__ == '__main__':
    import os

    root = '../data'
    bgls = [82, 96, 146, 180]
    n_samples = 12
    save = '../VAE'

    for bgl in bgls:
        folder = os.path.join(root, str(bgl))
        gt_folder = os.path.join(root, str(bgl))
        os.makedirs(os.path.join(save, str(bgl)), exist_ok=True)

        for sample in range(n_samples):
            Mura = load_img(os.path.join(folder, f'Mura_{sample+1}.png'))
            Mura = Mura.filter(ImageFilter.GaussianBlur(radius=11))
            Mura = np.array(Mura, dtype=np.float)
            Base, _ = fit_transform(X=Mura, bgl=bgl)
            save_img(f"{save}/{bgl}/Base_{sample+1:02d}.png", Base)

