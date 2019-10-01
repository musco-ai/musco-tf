# General utils for layer compressors.

import numpy as np
import tensorflow as tf
import sys
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat


def to_tf_kernel_order(tensor):
    """Change conv.kernel axis order from PyTorch to Tensoflow.

    :param tensor: tensor with conv.kernel weights.
    :return: tensor with the Tensoflow-like exis order.
    []
    """

    return np.transpose(tensor, (2, 3, 1, 0))


def to_pytorch_kernel_order(tensor):
    """Change conv.kernel axis order from Tensoflow to PyTorch.

    :param tensor: tensor with conv.kernel weights.
    :return: tensor with the Pytorch-like exis order.
    []
    """

    return np.transpose(tensor, (3, 2, 0, 1))


def del_keys(src_dict, del_keys):
    """Deletes redundant_keys from conf.

    :param src_dict: a dict
    :param del_keys: a list/set/etc with key names that we want to delete.
    :return: the copy of dict without keys from del_keys.
    """

    return {key: value for key, value in src_dict.items() if key not in del_keys}


def freeze_model(output_node_name, folder, pb_name):
    sess = tf.keras.backend.get_session()
    constant_graph = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(),
                                                                  [output_node_name])
    tf.train.write_graph(constant_graph, folder, pb_name, as_text=False)


def pb_to_tensorboard(model_filename, logdir_path):
    with tf.Session() as sess:
        with gfile.FastGFile(model_filename, "rb") as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)

            if 1 != len(sm.meta_graphs):
                print("More than one graph found. Not sure which to write")
                sys.exit(1)

            tf.import_graph_def(sm.meta_graphs[0].graph_def)

        train_writer = tf.summary.FileWriter(logdir_path)
        train_writer.add_graph(sess.graph)


def load_graph(graph_path, device=None):
    """Load frozen TensorFlow graph."""

    if device is not None:
        with tf.device(device):
            with tf.gfile.GFile(graph_path, "rb") as graph_file:
                graph_definition = tf.GraphDef()
                graph_definition.ParseFromString(graph_file.read())
    else:
        with tf.gfile.GFile(graph_path, "rb") as graph_file:
            graph_definition = tf.GraphDef()
            graph_definition.ParseFromString(graph_file.read())

    graph = tf.Graph()

    with graph.device(device):
        with graph.as_default():
            tf.import_graph_def(graph_definition, name="")

    return graph


def load_trt_graph(graph_path, device=None):
    """Load frozen TensorRT graph and fix device assignment."""

    if device is not None:
        with tf.device(device):
            with tf.gfile.GFile(graph_path, "rb") as graph_file:
                graph_definition = tf.GraphDef()
                graph_definition.ParseFromString(graph_file.read())
    else:
        with tf.gfile.GFile(graph_path, "rb") as graph_file:
            graph_definition = tf.GraphDef()
            graph_definition.ParseFromString(graph_file.read())

    graph = tf.Graph()

    with graph.device(device):
        with graph.as_default():
            tf.import_graph_def(graph_definition, name="")

    return graph
