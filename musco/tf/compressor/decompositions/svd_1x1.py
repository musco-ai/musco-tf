# Decompose 1x1 conv2d using SVD

import numpy as np

from tensorflow import keras

from musco.tf.compressor.decompositions.constructor import construct_compressor
from musco.tf.compressor.common.utils import del_keys
from musco.tf.compressor.rank_selection.estimator import estimate_rank_for_compression_rate, estimate_vbmf_ranks
from musco.tf.compressor.common.utils import to_tf_kernel_order, to_pytorch_kernel_order
from musco.tf.compressor.decompositions.svd import get_truncated_svd
from musco.tf.compressor.exceptions.compression_error import CompressionError


def get_params(layer):
    cin = None
    cout = None
    kernel_size = None
    padding = None
    strides = None
    activation = None
    batch_input_shape = None

    if isinstance(layer, keras.Sequential):
        # If the layer has been decomposed at least once, then
        # the first layer in a sequence contains in_channels,
        # the second layer contains information about kernel_size, padding and strides,
        # the third layer contains information about out_channels.
        layer_1, layer_2 = layer.layers
        conf_1, conf_2 = layer_1.get_config(), layer_2.get_config()

        if "batch_input_shape" in conf_1:
            batch_input_shape = conf_1["batch_input_shape"]

        cin = layer.input_shape[-1] if layer_1.data_format == "channels_last" else layer.input_shape[0]
        cout = layer.output_shape[-1] if layer_2.data_format == "channels_last" else layer.output_shape[0]
        kernel_size = conf_2["kernel_size"]
        padding = conf_2["padding"]
        strides = conf_2["strides"]
        activation = conf_2["activation"]
    elif isinstance(layer, keras.layers.Conv2D):
        cin = layer.input_shape[-1] if layer.data_format == "channels_last" else layer.input_shape[0]
        cout = layer.output_shape[-1] if layer.data_format == "channels_last" else layer.output_shape[0]
        layer_conf = layer.get_config()
        kernel_size = layer_conf["kernel_size"]
        padding = layer_conf["padding"]
        strides = layer_conf["strides"]
        activation = layer_conf["activation"]

        if "batch_input_shape" in layer_conf:
            batch_input_shape = layer_conf["batch_input_shape"]

    if cin is None or cout is None or kernel_size is None or padding is None or strides is None or \
            activation is None:
        raise CompressionError()

    return dict(cin=cin, cout=cout, kernel_size=kernel_size, padding=padding, strides=strides,
                batch_input_shape=batch_input_shape, activation=activation)


def get_rank(layer, rank, cin, cout,
             rank_selection="manual", vbmf_weaken_factor=1.0,
             param_reduction_rate=None, **kwargs):
    if rank_selection == 'vbmf':
        if isinstance(layer, keras.Sequential):
            return estimate_vbmf_ranks(to_pytorch_kernel_order(layer.get_weights()[1]), vbmf_weaken_factor)
        else:
            return estimate_vbmf_ranks(to_pytorch_kernel_order(layer.get_weights()[0]), vbmf_weaken_factor)
    elif rank_selection == 'manual':
        return int(rank)
    elif rank_selection == 'param_reduction':
        if isinstance(layer, keras.Sequential):
            return layer.layers[0].output_shape[-1] // param_reduction_rate
        else:
            return estimate_rank_for_compression_rate((cout, cin),
                                                      rate=param_reduction_rate,
                                                      key='svd')


def get_weights_and_bias(layer):
    """Returns weights and biases.

    :param layer: a source layer
    :return: If layer is tf.keras.layers.Conv2D layer.weights is returned as weights,
             Otherwise a list of weight tensors and bias tensor are returned as weights.
             The second element that is returned is a bias tensor.
             Note that all weights are returned in PyTorch dimension order:
             [out_channels, in_channels, kernel_size[0]*kernel_size[1]]
    """

    weights = None
    bias = None

    if isinstance(layer, keras.Sequential):
        weights, bias = layer.layers[-1].get_weights()
    elif isinstance(layer, keras.layers.Conv2D):
        weights, bias = layer.get_weights()

    weights = to_pytorch_kernel_order(weights)
    weights = weights.reshape(weights.shape[:2])

    if weights is None or bias is None:
        raise CompressionError()

    return weights, bias


def get_svd_factors(layer, rank, **kwargs):
    weights, bias = get_weights_and_bias(layer=layer)
    u, s, v_adj = get_truncated_svd(weights, rank)

    w0 = np.dot(np.sqrt(s), v_adj)
    w1 = np.dot(u, np.sqrt(s))
    if isinstance(layer, keras.Sequential):
        w0_old = layer.layers[0].get_weights()[0]
        w0 = np.dot(w0, w0_old)

    w0, w1 = [to_tf_kernel_order(w.reshape((*w.shape, 1, 1))) for w in [w0, w1]]

    return [w0, w1], [None, bias]


def get_layers_params_for_factors(cout, rank, kernel_size, padding, strides, batch_input_shape, activation, **kwargs):
    new_layers = [keras.layers.Conv2D, keras.layers.Conv2D]
    params = [
        dict(kernel_size=(1, 1), filters=rank, padding="same", use_bias=False),
        dict(kernel_size=kernel_size, filters=cout, padding=padding, strides=strides, activation=activation),
    ]

    if batch_input_shape is not None:
        params[0]["batch_input_shape"] = batch_input_shape

    return new_layers, params


def get_config(layer, copy_conf):
    confs = None
    if isinstance(layer, keras.Sequential):
        confs = [l.get_config() for l in layer.layers]
    elif isinstance(layer, keras.layers.Conv2D):
        if copy_conf:
            confs = [layer.get_config()] * 2
        else:
            confs = [{}] * 2

    redundant_keys = {"kernel_initializer", "bias_initializer", "name", "kernel_size", "padding", "strides", "filters",
                      "activation"}
    return [del_keys(conf, redundant_keys)for conf in confs]


get_svd_1x1_seq = construct_compressor(get_params, get_rank, get_svd_factors, get_layers_params_for_factors, get_config,
                                       (keras.layers.Conv2D, keras.Sequential))
