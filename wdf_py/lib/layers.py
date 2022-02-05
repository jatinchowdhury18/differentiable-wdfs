'''Custom layers for constructing differentiable WDF models'''

import numpy as np
import tensorflow as tf


class DenseLayer(tf.Module):
    """Dense layer without weights sharing"""

    def __init__(
        self,
        in_size,
        out_size,
        kernel_init=tf.keras.initializers.Orthogonal(),
        bias_init=tf.keras.initializers.Zeros(),
    ):
        super(DenseLayer, self).__init__()
        self.kernel = tf.Variable(
            self.init_weights(in_size, out_size, kernel_init), dtype=tf.float32
        )
        self.bias = tf.Variable(self.init_bias(out_size, bias_init), dtype=tf.float32)

    def init_weights(self, size1, size2, initializer):
        init = initializer(shape=(size1, size2))
        return [init]

    def init_bias(self, size, initializer):
        init = initializer(shape=(size))
        return [init]

    def set_weights(self, json_weights):
        weights = json_weights[0]
        self.kernel.assign(np.array([weights]))

        bias = json_weights[1]
        self.bias.assign(np.array([bias]))

    def __call__(self, input):
        return tf.matmul(input, self.kernel) + self.bias


class DenseRootModel(tf.Module):
    '''Simple Root WDF model using dense layers'''

    def __init__(self, json):
        super(DenseRootModel, self).__init__()

        self.a = tf.Variable(initial_value=tf.zeros(1), name="incident_wave")
        self.b = tf.Variable(initial_value=tf.zeros(1), name="reflected_wave")

        in_size = json["in_shape"][-1]
        layers_json = json["layers"]
        self.layers = []

        prev_size = in_size
        for l in layers_json:
            if l["type"] == "dense":
                next_size = l["shape"][-1]
                print(f"Adding Dense layer with size [{prev_size}, {next_size}]")

                self.layers.append(DenseLayer(prev_size, next_size))
                self.layers[-1].set_weights(l["weights"])
                prev_size = next_size

                if l["activation"] == "relu":
                    print("Adding ReLU activation layer")
                    self.layers.append(tf.nn.relu)
                elif l["activation"] == "tanh":
                    print("Adding tanh activation layer")
                    self.layers.append(tf.nn.tanh)

    def incident(self, x):
        self.a = x[:, :, 0]
        self.model_in = x

    def reflected(self):
        x = self.model_in
        for l in self.layers:
            x = l(x)

        self.b = -1 * x
        return self.b
