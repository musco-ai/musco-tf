import numpy as np
from sktensor import dtensor, tucker
from tensorflow import keras
from musco.tf.compressor.decompositions.constructor import construct_compressor
from musco.tf.compressor.rank_selection.estimator import estimate_rank_for_compression_rate, estimate_vbmf_ranks
from musco.tf.compressor.common.utils import to_tf_kernel_order, to_pytorch_kernel_order
from musco.tf.compressor.exceptions.compression_error import CompressionError


def get_conv_params(layer):
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
        layer_1, layer_2, layer_3 = layer.layers
        conf_1 = layer_1.get_config()
        conf_2 = layer_2.get_config()
        conf_3 = layer_3.get_config()

        if "batch_input_shape" in conf_1:
            batch_input_shape = conf_1["batch_input_shape"]

        cin = layer.input_shape[-1] if layer.layers[0].data_format == "channels_last" else layer.input_shape[0]
        cout = layer.output_shape[-1] if layer.layers[0].data_format == "channels_last" else layer.output_shape[0]
        kernel_size = conf_2["kernel_size"]
        padding = conf_2["padding"]
        strides = conf_2["strides"]
        activation = conf_3["activation"]
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
        weights = layer.layers[1].get_weights()[0]
        bias = layer.layers[-1].get_weights()[-1]
    elif isinstance(layer, keras.layers.Conv2D):
        weights, bias = layer.get_weights()

    weights = to_pytorch_kernel_order(weights)
    weights = weights.reshape((*weights.shape[:2], -1))

    if weights is None or bias is None:
        raise CompressionError()

    return weights, bias


def get_tucker_factors(layer, rank, cin, cout, kernel_size, **kwargs):
    weights, bias = get_weights_and_bias(layer)
    # print("Weights: ", weights.shape, "\nKernel Size: ", kernel_size)
    core, (U_cout, U_cin, U_dd) = tucker.hooi(dtensor(weights), [rank[0], rank[1], weights.shape[-1]], init="nvecs")
    core = core.dot(U_dd.T)
    w_cin = np.array(U_cin)
    w_core = np.array(core)
    w_cout = np.array(U_cout)

    if isinstance(layer, keras.Sequential):
        w_cin_old, w_cout_old = [to_pytorch_kernel_order(w) for w in [layer.layers[0].get_weights()[0],
                                                                      layer.layers[-1].get_weights()[0]]]

        U_cin_old = w_cin_old.reshape(w_cin_old.shape[:2]).T
        U_cout_old = w_cout_old.reshape(w_cout_old.shape[:2])
        w_cin = U_cin_old.dot(U_cin)
        w_cout = U_cout_old.dot(U_cout)

    # Reshape to the proper PyTorch shape order.
    w_cin = w_cin.T.reshape((rank[1], cin, 1, 1))
    w_core = w_core.reshape((rank[0], rank[1], *kernel_size))
    w_cout = w_cout.reshape((cout, rank[0], 1, 1))

    # Reorder to TensorFlow order.
    w_cin, w_core, w_cout = [to_tf_kernel_order(w) for w in [w_cin, w_core, w_cout]]

    return [w_cin, w_core, w_cout], [None, None, bias]


def get_layers_params_for_factors(cout, rank, kernel_size, padding, strides, batch_input_shape, activation, **kwargs):
    new_layers = [keras.layers.Conv2D, keras.layers.Conv2D, keras.layers.Conv2D]
    params = [
        dict(kernel_size=(1, 1), filters=rank[1], padding="same", use_bias=False),
        dict(kernel_size=kernel_size, filters=rank[0], padding=padding, strides=strides, use_bias=False),
        dict(kernel_size=(1, 1), padding="same", filters=cout, activation=activation)
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
            confs = [layer.get_config()] * 3
        else:
            confs = [{}] * 3

    # New layers have other "units", "kernel_initializer", "bias_initializer" and "name".
    # That's why we delete them to prevent double definition.
    for conf_idx, _ in enumerate(confs):
        for key in ["kernel_initializer", "bias_initializer", "name", "kernel_size", "padding", "strides", "filters",
                    "activation"]:
            if key not in confs[conf_idx]:
                continue

            del confs[conf_idx][key]

    return confs


def get_rank(layer, rank, cin, cout, kernel_size, vbmf=False, vbmf_weaken_factor=1.0, weight=None, **kwargs):
    if vbmf:
        if isinstance(layer, keras.Sequential):
            return estimate_vbmf_ranks(to_pytorch_kernel_order(layer.get_weights()[1]), vbmf_weaken_factor)
        else:
            return estimate_vbmf_ranks(to_pytorch_kernel_order(layer.get_weights()[0]), vbmf_weaken_factor)
    else:
        return estimate_rank_for_compression_rate((cout, cin, *kernel_size), rate=rank[0], key="tucker2")


get_tucker2_seq = construct_compressor(get_conv_params, get_rank, get_tucker_factors, get_layers_params_for_factors,
                                       get_config, (keras.layers.Conv2D, keras.Sequential))
