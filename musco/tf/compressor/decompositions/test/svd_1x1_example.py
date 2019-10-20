import numpy as np
import tensorflow as tf

from tensorflow import keras
from musco.tf.compressor.compress import compress_seq, compress_noseq


def test_tucker2(take_first=None):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if take_first is not None:
        train_images = train_images[:take_first, ...]
        train_labels = train_labels[:take_first, ...]

        test_images = test_images[:take_first, ...]
        test_labels = test_labels[:take_first, ...]

    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
    print (train_images[0].shape)
    def createModel() :
        input = tf.keras.layers.Input(shape=(28,28,1))
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   padding='valid',
                                   activation='relu',
                                   input_shape=(28, 28, 1))(input)
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   padding='valid',
                                   activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(1, 1),
                                   padding='valid',
                                   activation='relu',
                                   name='test_1')(x)
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(1, 1),
                                   padding='valid',
                                   activation='relu',
                                   name='test_2')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(10, activation='softmax')(x)
        return tf.keras.Model(input, x)
    model = createModel()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images,
              train_labels,
              epochs=2)

    model.summary()

    print('Evaluate source model')
    test_loss, test_acc = model.evaluate(test_images,
                                         test_labels,
                                         verbose=0)
    print('Test accuracy:', test_acc)

    compressed_model = compress_noseq(model, {
        'test_1': ('svd_1x1', 64),
        'test_2': ('svd_1x1', 50)
    })

    compressed_model.compile(optimizer='adam',
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])

    print('Evaluate compressed model')
    test_loss, test_acc = compressed_model.evaluate(test_images,
                                                    test_labels,
                                                    verbose=0)

    compressed_model.summary()
    print('Test accuracy:', test_acc)

    # for layer in compressed_model.layers:
    #     print(layer.name)


def test_tucker2_n_stages(take_first=None):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if take_first is not None:
        train_images = train_images[:take_first, ...]
        train_labels = train_labels[:take_first, ...]

        test_images = test_images[:take_first, ...]
        test_labels = test_labels[:take_first, ...]

    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
    def createModel():
        inputs = tf.keras.layers.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   padding='valid',
                                   activation='relu',
                                   input_shape=(28, 28, 1))(inputs)
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   padding='valid',
                                   activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(1, 1),
                                   padding='valid',
                                   activation='relu',
                                   name='test_1')(x)
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(1, 1),
                                   padding='valid',
                                   activation='relu',
                                   name='test_2')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(10, activation='softmax')(x)
        return tf.keras.Model(inputs, x)
    model = createModel()
    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images,
              train_labels,
              epochs=2)

    print('Evaluate source model')
    test_loss, test_acc = model.evaluate(test_images,
                                         test_labels,
                                         verbose=0)
    print('Test accuracy:', test_acc)

    compressed_model = model
    for idx in range(2):
        print("BEFORE COMPRESS")
        compressed_model.summary()
        compressed_model = compress_noseq(compressed_model, {
            'test_1': ('svd_1x1', 10),
            'test_2': ('svd_1x1', 10)
        })

        compressed_model.compile(optimizer='adam',
                                 loss='sparse_categorical_crossentropy',
                                 metrics=['accuracy'])

        print('Evaluate compressed model', idx + 1)
        test_loss, test_acc = compressed_model.evaluate(test_images,
                                                        test_labels,
                                                        verbose=0)

        # compressed_model.summary()
        print('Test accuracy:', test_acc)


# TODO: write regular tests
if __name__ == "__main__":
    print("!!!!", tf.__version__)
    #test_tucker2(10)
    test_tucker2_n_stages(1000)
