# %%
import sys
sys.path.insert(0, './lib')
sys.path.insert(0, './models')

import numpy as np
from scipy.special import wrightomega
import matplotlib.pyplot as plt
import pandas as pd

import tf_wdf as wdf
from tf_wdf import tf

from tqdm import tqdm
import pathlib
import json

from model_utils import *

# %%
BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
raw_data = pd.read_csv(f"{BASE_DIR}/diode_dataset/trial_data/47k_2.2nF_1N4148.csv", header=9)
raw_data = raw_data.to_numpy()

# %%
# input = [0], output [1]
FS = 50000
start = int(500e3)
N = int(20e3)
x = raw_data[start:start+N, 0].astype(np.float32)
R_data = np.ones_like(x) * 47e3
y_ref = raw_data[start:start+N, 1].astype(np.float32)

print(x.shape)
print(y_ref.shape)

# %%
batch_size = 2048
n_batches = N // batch_size

data_in = np.stack([x, R_data], axis=0).transpose()
data_in_trim = data_in[:(n_batches * batch_size), :]
data_in_batched = np.stack(np.array_split(data_in_trim, n_batches))

data_target = np.transpose(np.array([y_ref]))
data_target_trim = data_target[:(n_batches * batch_size), :]
data_target_batched = np.stack(np.array_split(data_target_trim, n_batches))

print(data_in.shape)
print(data_in_batched.shape)
print(data_target.shape)
print(data_target_batched.shape)

plot_batch = 5
# plt.plot(np.log(data_in_batched[plot_batch, :, 1]) - 10)
plt.plot(data_in_batched[plot_batch, :, 0])
plt.plot(data_target_batched[plot_batch, :, 0])

# %%
class DenseLayer(tf.Module):
    """ Dense layer without weights sharing"""
    def __init__(self, in_size, out_size):
        super(DenseLayer, self).__init__()
        self.kernel = tf.Variable(self.init_weights(in_size, out_size), dtype=tf.float32)
        self.bias = tf.Variable(self.init_bias(out_size), dtype=tf.float32)

    def init_weights(self, size1, size2):
        initializer = tf.keras.initializers.Orthogonal()
        init = initializer(shape=(size1, size2))
        return [init]

    def init_bias(self, size):
        initializer = tf.keras.initializers.Zeros()
        init = initializer(shape=(size))
        return [init]

    def set_weights(self, json_weights):
        weights = json_weights[0]
        self.kernel.assign(np.array([weights]))
    
        bias = json_weights[1]
        self.bias.assign(np.array([bias]))

    def __call__(self, input):
        return tf.matmul(input, self.kernel) + self.bias

class RootModel(tf.Module):
    def __init__(self, json):
        super(RootModel, self).__init__()

        self.a = tf.Variable(initial_value=tf.zeros(1), name='incident_wave')
        self.b = tf.Variable(initial_value=tf.zeros(1), name='reflected_wave')

        in_size = json['in_shape'][-1]
        layers_json = json['layers']
        self.layers = []

        prev_size = in_size
        for l in layers_json:
            if l['type'] == 'dense':
                next_size = l['shape'][-1]
                print(f'Adding Dense layer with size [{prev_size}, {next_size}]')

                self.layers.append(DenseLayer(prev_size, next_size))
                self.layers[-1].set_weights(l['weights'])
                prev_size = next_size

                if l['activation'] == 'relu':
                    print('Adding ReLU activation layer')
                    self.layers.append(tf.nn.relu)
                elif l['activation'] == 'tanh':
                    print('Adding tanh activation layer')
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

# %%
class ClipperModel(tf.Module):
    def __init__(self, json):
        super(ClipperModel, self).__init__()
        self.Vs = wdf.ResistiveVoltageSource(47000.0)
        self.C = wdf.Capacitor(2.2e-9, FS)
        self.P1 = wdf.Parallel(self.Vs, self.C)

        self.model = RootModel(json)

    def forward(self, input):
        sequence_length = input.shape[1]
        batch_size = input.shape[0]
        input = tf.cast(tf.expand_dims(input, axis=-1), dtype=tf.float32)
        output_sequence = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False)

        for i in range(sequence_length):
            self.Vs.set_voltage(input[:, i, 0:1])

            self.Vs.set_resistance(input[:, i, 1:2])
            self.P1.calc_impedance()

            model_in = tf.concat((self.P1.reflected(), tf.math.log(self.P1.R)), axis=1)
            self.model.incident(tf.transpose(model_in, perm=[0, 2, 1]))
            self.P1.incident(self.model.reflected())

            output = wdf.voltage(self.C)
            output_sequence = output_sequence.write(i, output)
        
        output_sequence = output_sequence.stack()
        return output_sequence

# %%
with open("./models/1N4148_pretrained_model.json", "r") as read_file:
    model_json = json.load(read_file)

model = ClipperModel(model_json)

# %%
def pre_emphasis_filter(x, coeff=0.85):
  return tf.concat([x[0:1], x[1:] - coeff*x[:-1]], axis=0)

eps = np.finfo(float).eps
def esr_loss(target_y, predicted_y, emphasis_func=lambda x : x):
    target_yp = emphasis_func(target_y)
    pred_yp = emphasis_func(predicted_y)
    mse = tf.math.reduce_sum(tf.math.square(target_yp - pred_yp))
    energy = tf.math.reduce_sum(tf.math.square(target_yp))
    
    loss_unnorm = mse / tf.cast(energy + eps, tf.float32)
    return tf.sqrt(loss_unnorm / N)

esr_with_emph = lambda target, pred: esr_loss(target, pred, pre_emphasis_filter)

def avg_loss(target_y, pred_y):
    target_mean = tf.math.reduce_mean(target_y)
    pred_mean = tf.math.reduce_mean(pred_y)
    return tf.math.abs(target_mean - pred_mean)

def bounds_loss(target_y, pred_y):
    target_min = tf.math.reduce_min(target_y)
    target_max = tf.math.reduce_max(target_y)
    pred_min = tf.math.reduce_min(pred_y)
    pred_max = tf.math.reduce_max(pred_y)
    return tf.math.abs(target_min - pred_min) + tf.math.abs(target_max - pred_max)

mse_loss = tf.keras.losses.MeanSquaredError()
# loss_func = lambda target, pred: 0.1 * esr_loss(target, pred) + 10 * mse_loss(target, pred)
loss_func = lambda target, pred: mse_loss(target, pred) + esr_loss(target, pred)

# optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-9)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
# optimizer = tf.keras.optimizers.Adagrad(learning_rate=1e-2, initial_accumulator_value=0.1, epsilon=1e-07,name='Adagrad')

# %%
for epoch in tqdm(range(501)):
# for epoch in range(101):
    with tf.GradientTape() as tape:
        outs = tf.transpose(model.forward(data_in_batched)[...,0], perm=[1, 0, 2])
        loss = loss_func(outs, data_target_batched)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 5 == 0:
        print(f'\nCheckpoint (Epoch = {epoch}):')
        print(f'    Loss: {loss}')
        plt.figure()
        plt.plot(data_target_batched[plot_batch, :, 0])
        plt.plot(outs[plot_batch, :, 0], '--')
        plt.savefig(f'./plots/scratch/1N4148_clipper_pot_epoch_{epoch}.png')
        plt.show()

print(f'\nFinal Results:')
print(f'    Loss: {loss}')

# %%
outs = tf.transpose(model.forward(data_in_batched)[...,0], perm=[1, 0, 2])

# %%
plt.plot(data_target_batched[plot_batch, :, 0])
plt.plot(outs[plot_batch, :, 0], '--')
# plt.ylim(-0.2, 0.2)
# plt.xlim(11150, 11900)

# %%
def save_model_json(model):
    def get_weights(layer):
        weights = layer.kernel.numpy()
        bias = layer.bias.numpy()
        return [weights, bias]

    model_dict = {}
    model_dict["in_shape"] = 2
    layers = []
    for layer in model.layers:
        if layer == tf.nn.tanh:
            layers[-1]["activation"] = 'tanh'
            continue
        
        if isinstance(layer, DenseLayer):
            layer_dict = {
                "type": 'dense',
                "shape": layer.bias.shape[-1],
                "weights": get_weights(layer)
            }
        layers.append(layer_dict)

    model_dict["layers"] = layers
    return model_dict

def save_model(model, filename):
    model_dict = save_model_json(model)
    with open(filename, 'w') as outfile:
        json.dump(model_dict, outfile, cls=NumpyArrayEncoder, indent=4)

save_model(model.model, './models/1N4148_clipper_pot.json')

# %%
