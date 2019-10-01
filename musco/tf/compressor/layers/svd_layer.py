"""
Compresses fully-connected layer using TruncatedSVD decomposition.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


def get_svd_seq(dense_layer, rank=None, copy_conf=False):
    """Returns a sequence of 3 fully connected layer that are equal to SVD decompositions of dense_layer.

    We decompose weights matrix of a fully-connected layer into three smaller matrices
    dense_layer = U*S*V.T, where
    - U has size [..., M, rank]
    - S has size [..., rank, rank]
    - V has size [..., N, rank]

    That is equal to a sequence of three dense layers with rank, rank and N units correspondingly.
    This method creates this layers and initializes them using U, S and V.

    :param dense_layer:
    :param rank: if it's not None launch TruncatedSVD, apply just SVD otherwise.
    :param copy_conf: whether or not copy conf of dense_layer into result's layers.
    :return: the sequence of three Dense layers [U, S, V.T] where dense_layer=U*S*V.T
    """

    weights, bias = dense_layer.get_weights()
    u, s, v_adj = np.linalg.svd(weights, full_matrices=False)

    # If rank is None take the original rank of weights.
    rank = min(weights.shape) if rank is None else rank

    # Truncate ranks.
    u = u[..., :rank]
    s = np.diag(s[..., :rank])
    v_adj = v_adj.T[..., :rank].T

    # Get conf of the source layer.
    confs = {}
    if copy_conf:
        confs = dense_layer.get_config()

        # New layers have other "units", 'kernel_initializer', 'bias_initializer" and "name'.
        # That's why we delete them to prevent double definition.
        del confs["units"], confs["kernel_initializer"], confs["bias_initializer"], confs["name"]

    # Make a sequence of 3 smaller dense layers.
    svd_seq = keras.Sequential([
        layers.Dense(units=rank,
                     kernel_initializer=tf.constant_initializer(u),
                     bias_initializer=tf.zeros_initializer(),
                     **confs),
        layers.Dense(units=rank,
                     kernel_initializer=tf.constant_initializer(s),
                     bias_initializer=tf.zeros_initializer(),
                     **confs),
        layers.Dense(units=weights.shape[-1],
                     kernel_initializer=tf.constant_initializer(v_adj),
                     bias_initializer=tf.constant_initializer(bias),  # add a source bias to the last layer
                     **confs),
    ])

    return svd_seq


class SVDLayer(layers.Layer):
    """
        This class compute the result using U, S, V -- the result of TruncatedSVD(src_matrix).
        If weights has size M x N then:
        - U has size [..., M, rank]
        - S has size [..., rank, rank]
        - V has size [..., N, rank]
        It adds an original bias vector after x*U*S*V.T computation to the result.
    """

    def __init__(self, dense_layer, rank=None, **kwargs):
        """ Returns a layer that is a result of SVD decompositions of "src_matrix".

        :param dense_layer: tf.keras.layers.Dense to decompose using TruncetedSVD
        :param rank: if it's not None launch TruncatedSVD, apply just SVD otherwise.
        """

        super(SVDLayer, self).__init__(**kwargs)
        weights, bias = dense_layer.get_weights()

        # Bias vector from the source fully-connected layer.
        self.bias = tf.Variable(initial_value=bias, name="bias")
        s, u, v = tf.linalg.svd(weights, full_matrices=False, compute_uv=True)

        # If rank is None take the original rank of weights.
        rank = min(weights.shape) if rank is None else rank

        # Truncate ranks
        u = u[..., :rank]
        s = s[..., :rank]
        v = v[..., :rank]
        s = tf.linalg.diag(s)

        # This variables will automatically be added to self.weights
        # in the order they are added below.
        # Refer https://www.tensorflow.org/beta/guide/keras/custom_layers_and_models#the_layer_class for details.
        self.u = tf.Variable(initial_value=u)
        self.s = tf.Variable(initial_value=s)
        self.v = tf.Variable(initial_value=v)

    def call(self, inputs, **kwargs):
        x = tf.matmul(inputs, self.u)
        x = tf.matmul(x, self.s)
        x = tf.matmul(x, self.v, adjoint_b=True)
        x = x + self.bias
        return x

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.v.get_shape()[0]
        return tuple(output_shape)
