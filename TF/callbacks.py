import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from PIL import Image


class SaveOutputImagesCallback(Callback):
    def __init__(
            self,
            t_model: tf.keras.Model,
            alg: str,
            save_dir: str,
            save_frequency: int,
            train_ds: tf.data.Dataset,
            test_ds: tf.data.Dataset
    ):
        super(SaveOutputImagesCallback, self).__init__()
        self.t_model = t_model
        self.alg = alg
        self.save_dir = save_dir
        self.save_frequency = save_frequency
        self.train_ds = train_ds
        self.test_ds = test_ds
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_frequency == 0:
            for i, data in enumerate(self.test_ds):
                if self.alg == 'vae':
                    output_images, _, _ = self.t_model.predict(data)
                else:
                    output_images = self.t_model.predict(data)

                for j, (y1, y2) in enumerate(zip(data, output_images)):
                    y1 = y1.numpy().squeeze()
                    y2 = y2.squeeze()

                    y1_path = os.path.join(
                        self.save_dir, f"INPUT_e{epoch + 1}_sample{j}.png")
                    y2_path = y1_path.replace('INPUT', 'TEST')
                    # img1 = Image.fromarray((y1*255).astype(np.uint8))
                    out = np.concatenate([y1, y2], axis=0)
                    img2 = Image.fromarray((out*255).astype(np.uint8))
                    # img1.save(y1_path)
                    img2.save(y2_path)


class SaveModelCallback(keras.callbacks.Callback):
    def __init__(self, t_model: tf.keras.Model, save_dir, save_name, save_frequency):
        super(SaveModelCallback, self).__init__()
        self.t_model = t_model
        self.save_dir = save_dir
        self.save_name = save_name
        self.save_frequency = save_frequency
        self.best = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_frequency == 0:
            model_path = os.path.join(self.save_dir, self.save_name)
            self.t_model.save_weights(model_path, save_format='h5')
            print(f"Model saved at epoch {epoch + 1} - {model_path}")


class CustomCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_batch_begin(self, batch, logs=None):
        lr = tf.keras.backend.get_value(
            self.model.optimizer.lr(self.model.optimizer.iterations))
        print(lr)
