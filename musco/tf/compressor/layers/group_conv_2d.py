import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn


class GroupConv2D(layers.Layer):
    def __init__(self,
                 rank,
                 n_group,
                 kernel_size,
                 strides=(1, 1),
                 padding="valid",
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=False,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupConv2D, self).__init__(**kwargs)
        self.rank = rank
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.n_group = n_group

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        channel_axis = -1
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim // self.n_group, self.rank)
        self.kernel = self.add_variable(name="kernel",
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_variable(name="bias",
                                          shape=(self.rank,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        groupConv = lambda i, k: tf.nn.conv2d(i,
                                              k,
                                              strides=[1, self.strides[0], self.strides[1], 1],
                                              padding=self.padding.upper())
        if self.n_group == 1:
            outputs = groupConv(inputs, self.kernel)
        else:
            inputGroups = tf.split(axis=3, num_or_size_splits=self.n_group, value=inputs)
            weightsGroups = tf.split(axis=3, num_or_size_splits=self.n_group, value=self.kernel)
            convGroups = [groupConv(i, k) for i, k in zip(inputGroups, weightsGroups)]
            outputs = tf.concat(axis=3, values=convGroups)

        if self.use_bias:
            if self.data_format != "channel_last" is None:
                raise ValueError("NHWC only.")
            outputs = nn.bias_add(outputs, self.bias, data_format="NHWC")

        if self.activation is not None:
            return self.activation(outputs)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()

        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)

        return tensor_shape.TensorShape([input_shape[0]] + new_space + [self.filters])

    def get_config(self):
        config = dict(n_group=self.n_group, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                      data_format=self.data_format, dilation_rate=self.dilation_rate,
                      activation=activations.serialize(self.activation), use_bias=self.use_bias,
                      kernel_initializer=initializers.serialize(self.kernel_initializer),
                      bias_initializer=initializers.serialize(self.bias_initializer),
                      kernel_regularizer=regularizers.serialize(self.kernel_regularizer),
                      bias_regularizer=regularizers.serialize(self.bias_regularizer),
                      activity_regularizer=regularizers.serialize(self.activity_regularizer),
                      kernel_constraint=constraints.serialize(self.kernel_constraint),
                      bias_constraint=constraints.serialize(self.bias_constraint))
        return config
