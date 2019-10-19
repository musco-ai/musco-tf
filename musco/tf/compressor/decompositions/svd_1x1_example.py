import numpy as np
import tensorflow as tf

from tensorflow import keras
from musco.tf.compressor.decompositions.svd_1x1 import get_svd_1x1_seq



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

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   padding='valid',
                                   activation='relu',
                                   input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   padding='valid',
                                   activation='relu',
                                   name='test'),
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(1, 1),
                                   padding='valid',
                                   activation='relu',
                                   name='test'),
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(1, 1),
                                   padding='valid',
                                   activation='relu',
                                   name='test'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ]
    )

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images,
              train_labels,
              epochs=1)

    model.summary()

    print('Evaluate source model')
    test_loss, test_acc = model.evaluate(test_images,
                                         test_labels,
                                         verbose=0)
    print('Test accuracy:', test_acc)

    # compressed_model = get_compressed_sequential(model, {
    #     'test': ('tucker2', (50, 50)),
    # })
    #
    # compressed_model.compile(optimizer='adam',
    #                          loss='sparse_categorical_crossentropy',
    #                          metrics=['accuracy'])
    #
    # print('Evaluate compressed model')
    # test_loss, test_acc = compressed_model.evaluate(test_images,
    #                                                 test_labels,
    #                                                 verbose=0)
    #
    # compressed_model.summary()
    # print('Test accuracy:', test_acc)
    #
    # for layer in compressed_model.layers:
    #     print(layer.name)


# TODO: write regular tests
if __name__ == "__main__":
    import pip

    # pip.main(['install', 'tensrflow==1.13.1'])
    print("!!!!", tf.__version__)
    # test_tucker2(1000)
    # test_tucker2_seq(1000)
    # test_tucker2_optimize_rank(1000)
    test_tucker2(1000)
    # test_resnet50()
    # test_resnet50_pretrained()
