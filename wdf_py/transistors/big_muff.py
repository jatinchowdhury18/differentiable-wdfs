# %%
import sys

sys.path.insert(0, '../lib')
sys.path.insert(0, './models')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile

import tf_wdf as wdf
from tf_wdf import tf
import tensorflow.keras.initializers as inits
from layers import DenseLayer

from tqdm import tqdm
from pathlib import Path
import json
import pickle

from model_utils import *

# %%
# model name here...

# %%
FS, raw_data = wavfile.read('./big_muff.wav')
raw_data = raw_data / 2**31
print(FS)

# %%
x = raw_data[:,0].astype(np.float32)
y_ref = raw_data[:,1].astype(np.float32) + 4.4
# y_ref = y_ref - np.mean(y_ref)

N = len(x)

print(x.shape)
print(y_ref.shape)

# %%
batch_size = 1500
n_batches = N // batch_size

data_in = np.stack([x], axis=0).transpose()
data_in_trim = data_in[:(n_batches * batch_size), :]
data_in_batched = np.stack(np.array_split(data_in_trim, n_batches))

data_target = np.transpose(np.array([y_ref]))
data_target_trim = data_target[:(n_batches * batch_size), :]
data_target_batched = np.stack(np.array_split(data_target_trim, n_batches))

print(data_in.shape)
print(data_in_batched.shape)
print(data_target.shape)
print(data_target_batched.shape)

plot_batch = 110
# plt.plot(np.log(data_in_batched[plot_batch, :, 1]) - 10)
plt.plot(data_in_batched[plot_batch, :, 0])
plt.plot(data_target_batched[plot_batch, :, 0])

# %%
class RModel(tf.Module):
    def __init__(self, NPorts, NInt, n_layers, layer_size):
        super(RModel, self).__init__()

        self.NPorts = NPorts
        self.NInt = NInt
        self.a = tf.Variable(initial_value=tf.zeros(NPorts), name='incident_waves')
        self.b = tf.Variable(initial_value=tf.zeros(NPorts), name='reflected_waves')

        self.state = tf.Variable(initial_value=tf.zeros(shape=(n_batches, 1, NInt)), name='model_state')

        self.layers = []
        prev_size = NInt
        for n in range(n_layers + 2):
            next_size = layer_size if n < n_layers + 1 else NInt
            print(f'Adding Dense layer with size [{prev_size}, {next_size}]')

            self.layers.append(DenseLayer(prev_size, next_size, inits.Zeros()))
            prev_size = next_size

            if n == n_layers + 1: continue
            print('Adding activation layer')
            self.layers.append(tf.nn.tanh)

    def set_S_data(self, Rs):
        Rd, Re, Rf, Rg, Rh = Rs

        Ra = Rb = Rc = 1.0e3 * tf.ones_like(Rs[0][0, 0, 0])
        Rd = Rd[0, 0, 0]
        Re = Re[0, 0, 0]
        Rf = Rf[0, 0, 0]
        Rg = Rg[0, 0, 0]
        Rh = Rh[0, 0, 0]

        S = np.array([ [ -((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra - Rb) * Rc + (Ra - Rb) * Rd) * Re + ((Ra - Rb) * Rc + (Ra - Rb) * Rd + (Ra - Rb) * Re) * Rf) * Rg - ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * ((Ra * Rc + Ra * Rd) * Re + (Ra * Rc + Ra * Rd + Ra * Re) * Rf) * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * Ra * Rb * Re * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * Ra * Rb * Re * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Rc + Ra * Rb * Rd) * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rc + Ra * Rd + Ra * Re) * Rf) * Rg) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh) ],
                       [ -2 * ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf) * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb - (Ra - Rb) * Rc - (Ra - Rb) * Rd) * Re - ((Ra - Rb) * Rc + (Ra - Rb) * Rd + (Ra - Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb - Rc - Rd) * Re - (Rc + Rd + Re) * Rf) * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Re * Rg + Rb * Re * Rg * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Re * Rg + Rb * Re * Rg * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * ((Rb * Rc + Rb * Rd) * Rg * Rh + (Ra * Rb * Rc + Ra * Rb * Rd) * Rg) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * ((Rb * Rc + Rb * Rd + Rb * Re) * Rg * Rh + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rg) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf) * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh) ],
                       [ 2 * Rb * Rc * Re * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rc * Re * Rg + Rc * Re * Rg * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -((Ra * Rb * Rc - Ra * Rb * Rd) * Re + (Ra * Rb * Rc - Ra * Rb * Rd - Ra * Rb * Re) * Rf + (Ra * Rb * Rc - Ra * Rb * Rd - (Ra * Rb - (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc - (Ra + Rb) * Rd - (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc - Rb * Rd) * Re + (Rb * Rc - Rb * Rd - Rb * Re) * Rf + (Rb * Rc - Rb * Rd - (Rb - Rc + Rd) * Re + (Rc - Rd - Re) * Rf) * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Rc * Re + Ra * Rb * Rc * Rf + (Ra * Rb * Rc + (Ra + Rb) * Rc * Re + (Ra + Rb) * Rc * Rf) * Rg + (Rb * Rc * Re + Rb * Rc * Rf + (Rb * Rc + Rc * Re + Rc * Rf) * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Rc * Rf + (Ra * Rb * Rc + (Ra + Rb) * Rc * Rf) * Rg + (Rb * Rc * Rf + (Rb * Rc + Rc * Rf) * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Rc * Re + (Ra + Rb) * Rc * Re * Rg + (Rb * Rc * Re + Rc * Re * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Rc * Re + Rb * Rc * Re * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * Rb * Rc * Re * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh) ],
                       [ 2 * Rb * Rd * Re * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rd * Re * Rg + Rd * Re * Rg * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Rd * Re + Ra * Rb * Rd * Rf + (Ra * Rb * Rd + (Ra + Rb) * Rd * Re + (Ra + Rb) * Rd * Rf) * Rg + (Rb * Rd * Re + Rb * Rd * Rf + (Rb * Rd + Rd * Re + Rd * Rf) * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), ((Ra * Rb * Rc - Ra * Rb * Rd) * Re + (Ra * Rb * Rc - Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc - Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc - (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc - (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc - Rb * Rd) * Re + (Rb * Rc - Rb * Rd + Rb * Re) * Rf + (Rb * Rc - Rb * Rd + (Rb + Rc - Rd) * Re + (Rc - Rd + Re) * Rf) * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Rd * Rf + (Ra * Rb * Rd + (Ra + Rb) * Rd * Rf) * Rg + (Rb * Rd * Rf + (Rb * Rd + Rd * Rf) * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Rd * Re + (Ra + Rb) * Rd * Re * Rg + (Rb * Rd * Re + Rd * Re * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Rd * Re + Rb * Rd * Re * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * Rb * Rd * Re * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh) ],
                       [ -2 * (Rb * Rc + Rb * Rd) * Re * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * ((Rc + Rd) * Re * Rg * Rh + (Ra * Rc + Ra * Rd) * Re * Rg) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Re * Rf + (Ra * Rb * Re + (Ra + Rb) * Re * Rf) * Rg + (Rb * Re * Rf + (Rb * Re + Re * Rf) * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Re * Rf + (Ra * Rb * Re + (Ra + Rb) * Re * Rf) * Rg + (Rb * Re * Rf + (Rb * Re + Re * Rf) * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -((Ra * Rb * Rc + Ra * Rb * Rd) * Re - (Ra * Rb * Rc + Ra * Rb * Rd - Ra * Rb * Re) * Rf - (Ra * Rb * Rc + Ra * Rb * Rd - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd - (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re - (Rb * Rc + Rb * Rd - Rb * Re) * Rf - (Rb * Rc + Rb * Rd - (Rb + Rc + Rd) * Re + (Rc + Rd - Re) * Rf) * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * (((Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re * Rg + (Ra * Rb * Rc + Ra * Rb * Rd) * Re + ((Rc + Rd) * Re * Rg + (Rb * Rc + Rb * Rd) * Re) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * ((Rb * Rc + Rb * Rd) * Re * Rh + (Ra * Rb * Rc + Ra * Rb * Rd) * Re) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Rb * Rc + Rb * Rd) * Re * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh) ],
                       [ 2 * (Rb * Rc + Rb * Rd + Rb * Re) * Rf * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * ((Rc + Rd + Re) * Rf * Rg * Rh + (Ra * Rc + Ra * Rd + Ra * Re) * Rf * Rg) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Re * Rf + (Ra + Rb) * Re * Rf * Rg + (Rb * Re * Rf + Re * Rf * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Re * Rf + (Ra + Rb) * Re * Rf * Rg + (Rb * Re * Rf + Re * Rf * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * (((Ra + Rb) * Rc + (Ra + Rb) * Rd) * Rf * Rg + (Ra * Rb * Rc + Ra * Rb * Rd) * Rf + ((Rc + Rd) * Rf * Rg + (Rb * Rc + Rb * Rd) * Rf) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), ((Ra * Rb * Rc + Ra * Rb * Rd) * Re - (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re - ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re - (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re - (Rc + Rd + Re) * Rf) * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * ((Rb * Rc + Rb * Rd + Rb * Re) * Rf * Rh + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * (Rb * Rc + Rb * Rd + Rb * Re) * Rf * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh) ],
                       [ -2 * ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf) * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * (((Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg * Rh + ((Ra * Rc + Ra * Rd) * Re + (Ra * Rc + Ra * Rd + Ra * Re) * Rf) * Rg) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Re * Rg + Rb * Re * Rg * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Re * Rg + Rb * Re * Rg * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * ((Rb * Rc + Rb * Rd) * Rg * Rh + (Ra * Rb * Rc + Ra * Rb * Rd) * Rg) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * ((Rb * Rc + Rb * Rd + Rb * Re) * Rg * Rh + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rg) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf - (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf - (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf) * Rg / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh) ],
                       [ -2 * ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * ((Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg * Rh / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * Rb * Re * Rg * Rh / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * Rb * Re * Rg * Rh / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * (Rb * Rc + Rb * Rd) * Rg * Rh / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), 2 * (Rb * Rc + Rb * Rd + Rb * Re) * Rg * Rh / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), -2 * ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf) * Rh / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh), ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg - ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh) / ((Ra * Rb * Rc + Ra * Rb * Rd) * Re + (Ra * Rb * Rc + Ra * Rb * Rd + Ra * Rb * Re) * Rf + (Ra * Rb * Rc + Ra * Rb * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + ((Ra + Rb) * Rc + (Ra + Rb) * Rd + (Ra + Rb) * Re) * Rf) * Rg + ((Rb * Rc + Rb * Rd) * Re + (Rb * Rc + Rb * Rd + Rb * Re) * Rf + (Rb * Rc + Rb * Rd + (Rb + Rc + Rd) * Re + (Rc + Rd + Re) * Rf) * Rg) * Rh) ]
            ])

        self.S11 = tf.convert_to_tensor(S[:self.NInt, :self.NInt])
        self.S22 = tf.convert_to_tensor(S[self.NInt:, self.NInt:])
        self.S12 = tf.transpose(tf.convert_to_tensor(S[:self.NInt, self.NInt:]))
        self.S21 = tf.transpose(tf.convert_to_tensor(S[self.NInt:, :self.NInt]))

    def incident(self, x):
        self.a = x[:, :, :]
        self.model_in = x

    def reflected(self):
        x = tf.matmul(self.model_in, self.S12) + tf.matmul(self.state, self.S11)
        for l in self.layers:
            x = l(x)
        self.state = self.state - x

        self.b = tf.matmul(self.state, self.S21) + tf.matmul(self.a, self.S22)
        return self.b

# %%
class BigMuffModel(tf.Module):
    def __init__(self):
        super(BigMuffModel, self).__init__()

        # Port D
        self.R21 = wdf.Resistor(150.0)

        # Port E
        self.VinR19 = wdf.ResistiveVoltageSource(10.0e3)
        self.C5 = wdf.Capacitor(100.0e-9, FS)
        self.S1_S2 = wdf.Series(self.VinR19, self.C5)

        self.R20 = wdf.Resistor(100.0e3)
        self.P2 = wdf.Parallel(self.R20, self.S1_S2)
        
        # Port F
        self.VccR18 = wdf.ResistiveVoltageSource(10.0e3)
        self.VccR18.set_voltage(9.0)

        # Port G
        self.R17 = wdf.Resistor(470.0e3)
        self.C12 = wdf.Capacitor(470.0e-12, FS)
        self.P1 = wdf.Parallel(self.R17, self.C12)

        # Port H
        self.C6 = wdf.Capacitor(1.0e-6, FS)

        self.r_facing_ports = [self.R21, self.P2, self.VccR18, self.P1, self.C6]

        self.model = RModel(5, 3, 2, 16)

    def forward(self, input):
        sequence_length = input.shape[1]
        input = tf.cast(tf.expand_dims(input, axis=-1), dtype=tf.float32)
        output_sequence = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False)

        model.state = tf.zeros(shape=(n_batches, 1, self.model.NInt))

        v_in = input[:, 0, 0:1]
        self.VinR19.set_voltage(tf.zeros_like(v_in))

        R_vals = []
        for p in self.r_facing_ports:
            p.calc_impedance()
            R_vals.append(p.R * tf.ones_like(v_in))

        self.model.set_S_data(R_vals)

        # run one-cycle to get vector sizes right...
        a_waves = []
        for p in self.r_facing_ports:
            a_waves.append(p.reflected())
            p.incident(tf.zeros_like(v_in))

        for i in range(sequence_length):
            self.VinR19.set_voltage(input[:, i, 0:1])

            for idx, p in enumerate(self.r_facing_ports):
                a_waves[idx] = p.reflected()

            model_in = tf.concat(a_waves, axis=1)
            self.model.incident(tf.transpose(model_in, perm=[0, 2, 1]))

            b_waves = self.model.reflected()
            for idx, p in enumerate(self.r_facing_ports):
                p.incident(b_waves[:, :, idx : idx + 1])

            output = wdf.voltage(self.VccR18)
            output_sequence = output_sequence.write(i, output)
        
        output_sequence = output_sequence.stack()
        return output_sequence

model = BigMuffModel()

# %%
def pre_emphasis_filter(x, coeff=0.85):
  return tf.concat([x[0:1], x[1:] - coeff*x[:-1]], axis=0)

def remove_dc_emph(x):
    x_mean = tf.math.reduce_mean(x)
    return x - x_mean

eps = np.finfo(float).eps
def esr_loss(target_y, predicted_y, emphasis_func=lambda x : x):
    target_yp = emphasis_func(target_y)
    pred_yp = emphasis_func(predicted_y)
    mse = tf.math.reduce_sum(tf.math.square(target_yp - pred_yp))
    energy = tf.math.reduce_sum(tf.math.square(target_yp))
    
    loss_unnorm = mse / tf.cast(energy + eps, tf.float32)
    return tf.sqrt(loss_unnorm / N)

def avg_loss(target_y, pred_y):
    target_mean = tf.math.reduce_mean(target_y)
    pred_mean = tf.math.reduce_mean(pred_y)
    return tf.math.abs(target_mean - pred_mean)

def bounds_loss(target_y, pred_y):
    target_min = tf.math.reduce_min(target_y, axis=1)
    target_max = tf.math.reduce_max(target_y, axis=1)
    pred_min = tf.math.reduce_min(pred_y, axis=1)
    pred_max = tf.math.reduce_max(pred_y, axis=1)

    b_diff = tf.math.abs(target_min - pred_min) + tf.math.abs(target_max - pred_max)
    return tf.math.reduce_mean(b_diff)

mse_loss = tf.keras.losses.MeanSquaredError()

# %%
def plot_target_pred(target, predicted, epoch):
    plt.figure()
    plt.plot(target[:batch_size], label='Target')
    plt.plot(predicted[:batch_size], '--', label='Predicted')
    plt.xlabel('Time [samples]')
    plt.ylabel('Voltage')
    plt.show()
    # plt.title(f'Diode Clipper ({diode_name}, {n_layers}x{layer_size}), Epoch {epoch}')
    # plt.legend(loc='lower left')

# %%
skip_samples = 750 # skip the first few samples to let state build up
history = { 'loss': [], 'mse': [], 'esr': [] }

# %%
def training_loop(num_epochs, optimizer, loss_func, history, plot_epochs=5):
    for epoch in tqdm(range(num_epochs)):
        with tf.GradientTape() as tape:
            outs = tf.transpose(model.forward(data_in_batched)[...,0], perm=[1, 0, 2])
            loss = loss_func(outs[:, skip_samples:, :], data_target_batched[:, skip_samples:, :])

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % plot_epochs == 0:
            print(f'\nCheckpoint (Epoch = {epoch}):')
            print(f'    Loss: {loss}')
            target = data_target_batched[plot_batch, skip_samples:, 0]
            pred = outs[plot_batch, skip_samples:, 0]
            plot_target_pred(target, pred, epoch)

        history['loss'].append(loss)
        history['mse'].append(mse_loss(outs[:, skip_samples:, :], data_target_batched[:, skip_samples:, :]))
        history['esr'].append(esr_loss(outs[:, skip_samples:, :], data_target_batched[:, skip_samples:, :]))

    print(f'\nFinal Results:')
    print(f'    Loss: {loss}')
    return history, outs

# %%
loss_func = mse_loss
optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-6)
history, outs = training_loop(101, optimizer, loss_func, history, plot_epochs=4)

# %%
# Run for 49 epochs with only AVG LOSS
loss_func = lambda target, pred: avg_loss(target, pred)
optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-3)
history, outs = training_loop(25, optimizer, loss_func, history, plot_epochs=3)

# %%
# Run for 50 epochs with MSE + AVG LOSS
loss_func = lambda target, pred: mse_loss(target, pred) + 100 * avg_loss(target, pred)
optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-4)
history, outs = training_loop(49, optimizer, loss_func, history, plot_epochs=4)

# %%
plot_batch2 = 80
input = data_in_batched[plot_batch2, skip_samples:, 0]
target = data_target_batched[plot_batch2, skip_samples:, 0]
pred = outs[plot_batch2, skip_samples:, 0]

plt.plot(target[:batch_size], label='Target')
plt.plot(pred[:batch_size], '--', label='Predicted')
# plt.plot(input[:batch_size], '--', label='Predicted')
 
# %%
loss_pred = outs[:, skip_samples:, :]
loss_target = data_target_batched[:, skip_samples:, :]

target_min = tf.math.reduce_min(loss_target, axis=1)
target_max = tf.math.reduce_max(loss_target, axis=1)
pred_min = tf.math.reduce_min(loss_pred, axis=1)
pred_max = tf.math.reduce_max(loss_pred, axis=1)

b_loss = tf.math.abs(target_min - pred_min) + tf.math.abs(target_max - pred_max)
print(tf.math.reduce_mean(b_loss))

# %%
plt.semilogy(history['mse'])

# %%
plt.semilogy(history['esr'])

# %%
# %%
def save_model_json(model):
    def get_weights(layer):
        weights = layer.kernel.numpy()[0]
        bias = layer.bias.numpy()[0]
        return [weights, bias]

    model_dict = {}
    model_dict["in_shape"] = (None, model.NInt)
    layers = []
    for layer in model.layers:
        if layer == tf.nn.tanh:
            layers[-1]["activation"] = 'tanh'
            continue
        
        if isinstance(layer, DenseLayer):
            layer_dict = {
                "type": 'dense',
                "shape": (None, layer.bias.shape[-1]),
                "weights": get_weights(layer),
                "activation": ""
            }
        layers.append(layer_dict)

    model_dict["layers"] = layers
    return model_dict

def save_model(model, filename):
    model_dict = save_model_json(model)
    with open(filename, 'w') as outfile:
        json.dump(model_dict, outfile, cls=NumpyArrayEncoder, indent=4)

save_model(model.model, f'./test_model.json')

# %%
