import tensorflow as tf
from model.GloRe import GloRe

def create_model(image_h, image_w, channel, n_class):
    inputs = tf.keras.layers.Input(shape=(image_h, image_w, channel))

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)  # 256*256*64

    x1 = tf.keras.layers.MaxPooling2D(padding='same')(x)  # 128*128*64

    x1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)  # 128*128*128

    x2 = tf.keras.layers.MaxPooling2D(padding='same')(x1)  # 64*64*128

    x2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)  # 64*64*256
    x2 = GloRe(x2, reduced_dim=128, num_node=80).forward(x2)

    x3 = tf.keras.layers.MaxPooling2D(padding='same')(x2)  # 32*32*256

    x3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)  # 32*32*512

    x4 = tf.keras.layers.MaxPooling2D(padding='same')(x3)  # 16*16*512

    x4 = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)  # 16*16*1024

    #  上采样部分

    x5 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2,
                                         padding='same', activation='relu')(x4)
    x5 = tf.keras.layers.BatchNormalization()(x5)  # 32*32*512

    x6 = tf.concat([x3, x5], axis=-1)  # 32*32*1024

    x6 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x6)
    x6 = tf.keras.layers.BatchNormalization()(x6)
    x6 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x6)
    x6 = tf.keras.layers.BatchNormalization()(x6)  # 32*32*512

    x7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2,
                                         padding='same', activation='relu')(x6)
    x7 = tf.keras.layers.BatchNormalization()(x7)  # 64*64*256

    x8 = tf.concat([x2, x7], axis=-1)  # 64*64*512

    x8 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x8)
    x8 = tf.keras.layers.BatchNormalization()(x8)
    x8 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x8)
    x8 = tf.keras.layers.BatchNormalization()(x8)  # 64*64*256

    x9 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2,
                                         padding='same', activation='relu')(x8)
    x9 = tf.keras.layers.BatchNormalization()(x9)  # 128*128*128

    x10 = tf.concat([x1, x9], axis=-1)  # 128*128*256

    x10 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x10)
    x10 = tf.keras.layers.BatchNormalization()(x10)
    x10 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x10)
    x10 = tf.keras.layers.BatchNormalization()(x10)  # 128*128*128

    x11 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2,
                                          padding='same', activation='relu')(x10)
    x11 = tf.keras.layers.BatchNormalization()(x11)  # 256*256*64

    x12 = tf.concat([x, x11], axis=-1)  # 256*256*128

    x12 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x12)
    x12 = tf.keras.layers.BatchNormalization()(x12)
    x12 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x12)
    x12 = tf.keras.layers.BatchNormalization()(x12)  # 256*256*64

    output = tf.keras.layers.Conv2D(n_class, 1, padding='same')(x12)
    #  256*256*34
    return tf.keras.Model(inputs=inputs, outputs=output)
