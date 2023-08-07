import os
import tensorflow as tf
import numpy as np


AUTO = tf.data.experimental.AUTOTUNE


def crop_image(image, crop_size, num_crops):
    crops = [tf.image.random_crop(
        image, size=(crop_size, crop_size, 1), seed=4715) for _ in range(num_crops)]
    return tf.stack(crops, axis=0)


def multi_crop_dataset(folder_path, batch_size, crop_size, num_crops):
    image_files = tf.data.Dataset.list_files(os.path.join(folder_path, '*'))

    def load_and_preprocess_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
        # image = tf.image.random_crop(image, size=(512, 512, 1), seed=4715)
        return image

    dataset = image_files.map(lambda x: load_and_preprocess_image(x))
    dataset = dataset.map(lambda x: crop_image(x, crop_size, num_crops),
                          num_parallel_calls=AUTO)
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    return dataset.shuffle(1000).batch(batch_size * num_crops).prefetch(AUTO)


def gaussian_blur(img, ksize=5, sigma=1):
    def gaussian_kernel(size=3, sigma=1):
        x_range = tf.range(-(size - 1) // 2, (size - 1) // 2 + 1, 1)
        y_range = tf.range((size - 1) // 2, -(size - 1) // 2 - 1, -1)

        xs, ys = tf.meshgrid(x_range, y_range)
        kernel = tf.exp(-(xs ** 2 + ys ** 2) / (2 * (sigma ** 2))) / (
                    2 * np.pi * (sigma ** 2))
        return tf.cast(kernel / tf.reduce_sum(kernel), tf.float32)

    kernel = gaussian_kernel(ksize, sigma)
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)

    r, g, b = tf.split(img, [1, 1, 1], axis=-1)
    r_blur = tf.nn.conv2d([r], kernel, [1, 1, 1, 1], 'SAME')
    g_blur = tf.nn.conv2d([g], kernel, [1, 1, 1, 1], 'SAME')
    b_blur = tf.nn.conv2d([b], kernel, [1, 1, 1, 1], 'SAME')

    blur_image = tf.concat([r_blur, g_blur, b_blur], axis=-1)
    return tf.squeeze(blur_image, axis=0)


def gaussian_blur_1c(img, ksize=5, sigma=1):
    def gaussian_kernel(size=3, sigma=1):
        x_range = tf.range(-(size - 1) // 2, (size - 1) // 2 + 1, 1)
        y_range = tf.range((size - 1) // 2, -(size - 1) // 2 - 1, -1)

        xs, ys = tf.meshgrid(x_range, y_range)
        kernel = tf.exp(-(xs ** 2 + ys ** 2) / (2 * (sigma ** 2))) / (
                    2 * np.pi * (sigma ** 2))
        return tf.cast(kernel / tf.reduce_sum(kernel), tf.float32)

    kernel = gaussian_kernel(ksize, sigma)
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], 'SAME')


def batch_dataset(folder_path, batch_size, training=True):
    image_files = tf.data.Dataset.list_files(os.path.join(folder_path, '*.png'))

    def load_and_preprocess_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, dtype=tf.float32)
        # image = gaussian_blur_1c(image, 11, 1)
        # image = tf.squeeze(image)
        # image = tf.image.resize_with_pad(image, 256, 256)
        return image

    dataset = image_files.map(lambda x: load_and_preprocess_image(x))

    # def apply_gaussian_noise(image_dataset, mean=0.0, stddev=1.0):
    #     noisy_dataset = image_dataset.map(
    #         lambda x: x + tf.random.normal(shape=tf.shape(x), mean=mean, stddev=stddev))
    #     return noisy_dataset
    # dataset = apply_gaussian_noise(dataset, mean=0.0, stddev=0.1)
    # dataset = dataset.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    if training:
        return dataset.batch(batch_size).shuffle(buffer_size=500).prefetch(AUTO)
    return dataset.batch(12).prefetch(AUTO)
