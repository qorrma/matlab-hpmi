from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC
import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), stride=(1, 1)):
        super(ResidualBlock, self).__init__()
        self.residual_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   strides=stride, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   strides=stride, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])

    def call(self, x, **kwargs):
        return x + self.residual_block(x)


class ResNetEncoder(tf.keras.models.Model, ABC):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=10,
                 bUseMultiResSkips=True):
        super(ResNetEncoder, self).__init__()
        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips
        self.conv_list = []
        self.res_blk_list = []
        self.multi_res_skip_list = []
        self.input_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3),
                                   strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2)
        ])

        for i in range(n_levels):
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (n_levels - i)

            self.res_blk_list.append(
                tf.keras.Sequential([ResidualBlock(n_filters_1)
                                     for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters=n_filters_2, kernel_size=(2, 2),
                                           strides=(2, 2), padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                ])
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    tf.keras.Sequential([
                        tf.keras.layers.Conv2D(filters=self.max_filters, kernel_size=(ks, ks),
                                               strides=(ks, ks), padding='same'),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(alpha=0.2),
                    ])
                )

        self.output_conv = tf.keras.layers.Conv2D(filters=z_dim, kernel_size=(3, 3),
                                                  strides=(1, 1), padding='same')

    def call(self, x, **kwargs):
        x = self.input_conv(x)
        skips = []
        for i in range(self.n_levels):
            x = self.res_blk_list[i](x)
            if self.bUseMultiResSkips:
                skips.append(self.multi_res_skip_list[i](x))
            x = self.conv_list[i](x)
        if self.bUseMultiResSkips:
            x = sum([x] + skips)
        x = self.output_conv(x)
        return x


class ResNetDecoder(tf.keras.models.Model, ABC):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 output_channels=3,
                 bUseMultiResSkips=True):
        super(ResNetDecoder, self).__init__()

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = []
        self.res_blk_list = []
        self.multi_res_skip_list = []

        self.input_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.max_filters, kernel_size=(3, 3),
                                   strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])

        for i in range(n_levels):
            n_filters = 2 ** (self.n_levels - i + 2)
            ks = 2 ** (i + 1)
            self.res_blk_list.append(
                tf.keras.Sequential([ResidualBlock(n_filters)
                                     for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                tf.keras.Sequential([
                    tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
                    tf.keras.layers.Conv2D(filters=n_filters,
                                           kernel_size=(3, 3),
                                           strides=(1, 1), padding='same'),
                    # tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=(2, 2),
                    #                                 strides=(2, 2), padding='same', ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                ])
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    tf.keras.Sequential([
                        # tf.keras.layers.UpSampling2D(size=(ks, ks)),
                        # tf.keras.layers.Conv2D(filters=n_filters,
                        #                        kernel_size=(3, 3),
                        #                        strides=(1, 1), padding='same'),
                        tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=(ks, ks),
                                                        strides=(ks, ks), padding='same'),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(alpha=0.2),
                    ])
                )

        self.output_conv = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=(3, 3),
                                                  strides=(1, 1), padding='same')

    def call(self, z, **kwargs):
        z = z_top = self.input_conv(z)
        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)
            if self.bUseMultiResSkips:
                z += self.multi_res_skip_list[i](z_top)
        z = self.output_conv(z)
        return z