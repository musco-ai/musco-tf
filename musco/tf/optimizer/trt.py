import os
import gc
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import graph_io


class Optimizer:
    def __init__(self, precision="FP16", max_batch_size=16):
        self.precision = precision
        self.max_batch_size = max_batch_size

    @staticmethod
    def freeze(model):
        outputs = [output.name.split(":")[0] for output in model.outputs]
        sess = tf.keras.backend.get_session()
        constant_graph = tf.graph_util.convert_variables_to_constants(sess,
                                                                      sess.graph.as_graph_def(),
                                                                      outputs)
        tf.train.write_graph(constant_graph, ".", "frozen.pb", as_text=False)
        print("Saved as frozen.pb.")

    def optimize(self, graph_path, output_names=None):
        name, extension = os.path.splitext(os.path.basename(graph_path))
        dirname = os.path.dirname(graph_path)
        print("Start optimizing {0}...".format(dirname.split(os.sep)[-1]))
        fixed_graph = tf.GraphDef()

        with tf.gfile.GFile(graph_path, "rb") as graph_file:
            fixed_graph.ParseFromString(graph_file.read())

        if output_names is None:
            output_names = [fixed_graph.node[-1].name]
        else:
            assert output_names[0] in [i.name for i in fixed_graph.node]

        print(output_names)

        quantized_graph = trt.create_inference_graph(
            input_graph_def=fixed_graph,
            outputs=output_names,
            max_batch_size=self.max_batch_size,
            max_workspace_size_bytes=1 << 32,
            precision_mode=self.precision)

        graph_io.write_graph(quantized_graph, dirname, name + "_trt" + extension, as_text=False)
        trt_engine_op = len([1 for n in quantized_graph.node if str(n.op) == "TRTEngineOp"])
        print("TRTEngineOp: {0}.".format(trt_engine_op))

        if trt_engine_op == 0:
            print("TensorRT failed to optimize graph.")

        print("Optimization for {0} finished.".format(dirname.split(os.sep)[-1]))
        del fixed_graph
        del quantized_graph
        gc.collect()
