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
from layers import DenseLayer

from tqdm import tqdm
from pathlib import Path
import json
import pickle

from model_utils import *

# %%
# model name here...

# %%
FS, raw_data = wavfile.read('./voltages.wav')
raw_data = raw_data / 2**31

# %%
x = raw_data[:,0].astype(np.float32)
y_ref = 5 * raw_data[:,1].astype(np.float32)

N = len(x)

print(x.shape)
print(y_ref.shape)

# %%
batch_size = 1024
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

plot_batch = 0
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

            self.layers.append(DenseLayer(prev_size, next_size))
            prev_size = next_size

            if n == 0 or n == n_layers + 1: continue
            print('Adding activation layer')
            self.layers.append(tf.nn.tanh)

    def set_S_data(self, Rs):
        Rc, Rd, Re, Rf, Rg, Rh = Rs

        Rc = Rc[0, 0, 0]
        Rd = Rd[0, 0, 0]
        Re = Re[0, 0, 0]
        Rf = Rf[0, 0, 0]
        Rg = Rg[0, 0, 0]
        Rh = Rh[0, 0, 0]
        Ra = Rb = 1.0e3 * tf.ones_like(Rc)

        S = np.array([
            [ -((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra - Rb - Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra - Rb - Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra - Rb - Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra - Rb - Rc) * Rd - (Rb + Rc + Rd) * Re) * Rf - ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * (Ra * Rd * Re * Rf + Ra * Rd * Rf * Rh + (Ra * Rd * Re + Ra * Rd * Rf) * Rg) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * (Ra * Rd * Re * Rf + Ra * Rd * Rf * Rh + (Ra * Rd * Re + Ra * Rd * Rf) * Rg) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Ra * Rb + Ra * Rc) * Re * Rf + (Ra * Rb + Ra * Rc) * Rf * Rh + ((Ra * Rb + Ra * Rc) * Re + (Ra * Rb + Ra * Rc) * Rf) * Rg) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * ((Ra * Rb + Ra * Rc) * Rd * Rg - (Ra * Rb + Ra * Rc + Ra * Rd) * Rf * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc) * Rd * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Ra * Rb + Ra * Rc) * Rd * Re + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + Ra * Rd) * Rf) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Ra * Rb + Ra * Rc + Ra * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + Ra * Rd) * Rf) * Rg) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh)], \
            [ -2 * (Rb * Rd * Re * Rf + Rb * Rd * Rf * Rh + (Rb * Rd * Re + Rb * Rd * Rf) * Rg) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -((Ra * Rb - Ra * Rc) * Rd * Re + (Ra * Rb - Ra * Rc - (Ra - Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb - Ra * Rc) * Rd + (Ra * Rb - Ra * Rc - (Ra - Rb + Rc) * Rd) * Re + (Ra * Rb - Ra * Rc - (Ra - Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb - Ra * Rc) * Rd + (Ra * Rb - Ra * Rc - Ra * Rd) * Re + (Ra * Rb - Ra * Rc - (Ra - Rb + Rc) * Rd + (Rb - Rc - Rd) * Re) * Rf + ((Rb - Rc) * Rd + (Rb - Rc - Rd) * Re + (Rb - Rc - Rd) * Rf) * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Rd * Re + (Ra * Rb + Rb * Rd) * Re * Rf + (Ra * Rb * Rd + (Ra * Rb + Rb * Rd) * Re + (Ra * Rb + Rb * Rd) * Rf) * Rg + (Ra * Rb * Rd + Ra * Rb * Re + (Ra * Rb + Rb * Rd + Rb * Re) * Rf + (Rb * Rd + Rb * Re + Rb * Rf) * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * (Ra * Rb * Re * Rf + (Ra * Rb * Re + Ra * Rb * Rf) * Rg + (Ra * Rb * Re + (Ra * Rb + Rb * Re) * Rf + (Rb * Re + Rb * Rf) * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Rd * Rg + (Ra * Rb * Rd + Rb * Rd * Rf + Rb * Rd * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * (Ra * Rb * Rd * Re + Ra * Rb * Rd * Rg + (Ra * Rb * Rd + Rb * Rd * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * (Ra * Rb * Rd * Re - Rb * Rd * Rf * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * (Ra * Rb * Rd * Re + Rb * Rd * Re * Rf + (Rb * Rd * Re + Rb * Rd * Rf) * Rg) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh)], \
            [ -2 * (Rc * Rd * Re * Rf + Rc * Rd * Rf * Rh + (Rc * Rd * Re + Rc * Rd * Rf) * Rg) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * (Ra * Rc * Rd * Re + (Ra * Rc + Rc * Rd) * Re * Rf + (Ra * Rc * Rd + (Ra * Rc + Rc * Rd) * Re + (Ra * Rc + Rc * Rd) * Rf) * Rg + (Ra * Rc * Rd + Ra * Rc * Re + (Ra * Rc + Rc * Rd + Rc * Re) * Rf + (Rc * Rd + Rc * Re + Rc * Rf) * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), ((Ra * Rb - Ra * Rc) * Rd * Re + (Ra * Rb - Ra * Rc + (Ra + Rb - Rc) * Rd) * Re * Rf + ((Ra * Rb - Ra * Rc) * Rd + (Ra * Rb - Ra * Rc + (Ra + Rb - Rc) * Rd) * Re + (Ra * Rb - Ra * Rc + (Ra + Rb - Rc) * Rd) * Rf) * Rg + ((Ra * Rb - Ra * Rc) * Rd + (Ra * Rb - Ra * Rc + Ra * Rd) * Re + (Ra * Rb - Ra * Rc + (Ra + Rb - Rc) * Rd + (Rb - Rc + Rd) * Re) * Rf + ((Rb - Rc) * Rd + (Rb - Rc + Rd) * Re + (Rb - Rc + Rd) * Rf) * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * (Ra * Rc * Re * Rf + (Ra * Rc * Re + Ra * Rc * Rf) * Rg + (Ra * Rc * Re + (Ra * Rc + Rc * Re) * Rf + (Rc * Re + Rc * Rf) * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * (Ra * Rc * Rd * Rg + (Ra * Rc * Rd + Rc * Rd * Rf + Rc * Rd * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * (Ra * Rc * Rd * Re + Ra * Rc * Rd * Rg + (Ra * Rc * Rd + Rc * Rd * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * (Ra * Rc * Rd * Re - Rc * Rd * Rf * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * (Ra * Rc * Rd * Re + Rc * Rd * Re * Rf + (Rc * Rd * Re + Rc * Rd * Rf) * Rg) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh)], \
            [ -2 * ((Rb + Rc) * Rd * Re * Rf + (Rb + Rc) * Rd * Rf * Rh + ((Rb + Rc) * Rd * Re + (Rb + Rc) * Rd * Rf) * Rg) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * (Ra * Rd * Re * Rf + (Ra * Rd * Re + Ra * Rd * Rf) * Rg + (Ra * Rd * Re + (Ra * Rd + Rd * Re) * Rf + (Rd * Re + Rd * Rf) * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * (Ra * Rd * Re * Rf + (Ra * Rd * Re + Ra * Rd * Rf) * Rg + (Ra * Rd * Re + (Ra * Rd + Rd * Re) * Rf + (Rd * Re + Rd * Rf) * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -((Ra * Rb + Ra * Rc) * Rd * Re - (Ra * Rb + Ra * Rc - (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd - (Ra * Rb + Ra * Rc - (Ra + Rb + Rc) * Rd) * Re - (Ra * Rb + Ra * Rc - (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd - (Ra * Rb + Ra * Rc - Ra * Rd) * Re - (Ra * Rb + Ra * Rc - (Ra + Rb + Rc) * Rd + (Rb + Rc - Rd) * Re) * Rf + ((Rb + Rc) * Rd - (Rb + Rc - Rd) * Re - (Rb + Rc - Rd) * Rf) * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Ra * Rb + Ra * Rc) * Rd * Rg + ((Rb + Rc) * Rd * Rf + (Rb + Rc) * Rd * Rg + (Ra * Rb + Ra * Rc) * Rd) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc) * Rd * Rg + ((Rb + Rc) * Rd * Rg + (Ra * Rb + Ra * Rc) * Rd) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Rb + Rc) * Rd * Rf * Rh - (Ra * Rb + Ra * Rc) * Rd * Re) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Rb + Rc) * Rd * Re * Rf + (Ra * Rb + Ra * Rc) * Rd * Re + ((Rb + Rc) * Rd * Re + (Rb + Rc) * Rd * Rf) * Rg) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh)], \
            [ 2 * ((Rb + Rc) * Rd * Re * Rg - (Rb + Rc + Rd) * Re * Rf * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * (Ra * Rd * Re * Rg + (Ra * Rd * Re + Rd * Re * Rf + Rd * Re * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * (Ra * Rd * Re * Rg + (Ra * Rd * Re + Rd * Re * Rf + Rd * Re * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Ra * Rb + Ra * Rc) * Re * Rg + ((Rb + Rc) * Re * Rf + (Rb + Rc) * Re * Rg + (Ra * Rb + Ra * Rc) * Re) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf - ((Ra * Rb + Ra * Rc) * Rd - (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg - ((Ra * Rb + Ra * Rc) * Rd - (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd - (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd - (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * ((Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rg + ((Rb + Rc + Rd) * Re * Rg + (Ra * Rb + Ra * Rc + Ra * Rd) * Re) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Rb + Rc + Rd) * Re * Rf * Rh + (Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * ((Rb + Rc) * Rd * Re * Rg + (Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh)], \
            [ -2 * ((Rb + Rc) * Rd * Re * Rf + (Rb + Rc) * Rd * Rf * Rg + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * (Ra * Rd * Re * Rf + Ra * Rd * Rf * Rg + (Ra * Rd * Rf + Rd * Rf * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * (Ra * Rd * Re * Rf + Ra * Rd * Rf * Rg + (Ra * Rd * Rf + Rd * Rf * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * ((Ra * Rb + Ra * Rc) * Re * Rf + (Ra * Rb + Ra * Rc) * Rf * Rg + ((Rb + Rc) * Rf * Rg + (Ra * Rb + Ra * Rc) * Rf) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * ((Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf * Rg + ((Rb + Rc + Rd) * Rf * Rg + (Ra * Rb + Ra * Rc + Ra * Rd) * Rf) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), ((Ra * Rb + Ra * Rc) * Rd * Re - (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re - (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re - (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re - (Rb + Rc + Rd) * Rf) * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Rb + Rc) * Rd * Rf * Rg - (Ra * Rb + Ra * Rc + Ra * Rd) * Re * Rf) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh)], \
            [ -2 * ((Rb + Rc) * Rd * Re * Rg + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * (Ra * Rd * Re * Rg - Rd * Rf * Rg * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * (Ra * Rd * Re * Rg - Rd * Rf * Rg * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Rb + Rc) * Rf * Rg * Rh - (Ra * Rb + Ra * Rc) * Re * Rg) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Rb + Rc + Rd) * Rf * Rg * Rh + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rg + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rg * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf - ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf - ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh)], \
            [ -2 * ((Rb + Rc + Rd) * Re * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * (Ra * Rd * Re + Rd * Re * Rf + (Rd * Re + Rd * Rf) * Rg) * Rh / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * (Ra * Rd * Re + Rd * Re * Rf + (Rd * Re + Rd * Rf) * Rg) * Rh / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Rb + Rc) * Re * Rf + (Ra * Rb + Ra * Rc) * Re + ((Rb + Rc) * Re + (Rb + Rc) * Rf) * Rg) * Rh / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * ((Rb + Rc) * Rd * Rg + (Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rh / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), -2 * ((Rb + Rc) * Rd * Rg - (Ra * Rb + Ra * Rc + Ra * Rd) * Re) * Rh / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), 2 * ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rh / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh), ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg - ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh) / ((Ra * Rb + Ra * Rc) * Rd * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re * Rf + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd) * Rf) * Rg + ((Ra * Rb + Ra * Rc) * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re + (Ra * Rb + Ra * Rc + (Ra + Rb + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf + ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Rg) * Rh)]
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
        self.state = x

        self.b = tf.matmul(x, self.S21) + tf.matmul(self.a, self.S22)
        return self.b

# %%
class CommonEmitterModel(tf.Module):
    def __init__(self):
        super(CommonEmitterModel, self).__init__()

        # Port C
        self.Remitter = wdf.Resistor(220)
        self.Ce = wdf.Capacitor(100.0e-6, FS)
        self.Pc = wdf.Parallel(self.Remitter, self.Ce)

        # Port D
        self.VinRin = wdf.ResistiveVoltageSource(1.0e3)
        self.Cin = wdf.Capacitor(50.0e-6, FS)
        self.Sin = wdf.Series(self.VinRin, self.Cin)
        self.R2 = wdf.Resistor(2.65e3)
        self.Pd = wdf.Parallel(self.Sin, self.R2)

        # Port E
        self.B1 = wdf.ResistiveVoltageSource()
        self.B1.set_voltage(18)

        # Port F
        self.R1 = wdf.Resistor(27.35e3)

        # Port G
        self.Rcollector = wdf.Resistor(1.78e3)

        # Port H
        self.RL = wdf.Resistor(1.0e3)
        self.C2 = wdf.Capacitor(10.0e-6, FS)
        self.Sh = wdf.Series(self.RL, self.C2)

        self.model = RModel(6, 2, 4, 32)

    def forward(self, input):
        sequence_length = input.shape[1]
        input = tf.cast(tf.expand_dims(input, axis=-1), dtype=tf.float32)
        output_sequence = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False)

        self.Ce.reset()
        self.Cin.reset()
        self.C2.reset()

        v_in = input[:, 0, 0:1]
        self.VinRin.set_voltage(v_in)

        self.Pc.calc_impedance()
        self.Pd.calc_impedance()
        self.B1.calc_impedance()
        self.R1.calc_impedance()
        self.Rcollector.calc_impedance()
        self.Sh.calc_impedance()

        Rc = tf.math.log(self.Pc.R) * tf.ones_like(v_in)
        Rd = tf.math.log(self.Pd.R) * tf.ones_like(v_in)
        Re = tf.math.log(self.B1.R) * tf.ones_like(v_in)
        Rf = tf.math.log(self.R1.R) * tf.ones_like(v_in)
        Rg = tf.math.log(self.Rcollector.R) * tf.ones_like(v_in)
        Rh = tf.math.log(self.Sh.R) * tf.ones_like(v_in)

        self.model.set_S_data([Rc, Rd, Re, Rf, Rg, Rh])

        a_c = self.Pc.reflected()
        a_d = self.Pd.reflected()
        a_e = self.B1.reflected()
        a_f = self.R1.reflected()
        a_g = self.Rcollector.reflected()
        a_h = self.Sh.reflected()
        self.Pc.incident(tf.zeros_like(v_in))
        self.Pd.incident(tf.zeros_like(v_in))
        self.B1.incident(tf.zeros_like(v_in))
        self.R1.incident(tf.zeros_like(v_in))
        self.Rcollector.incident(tf.zeros_like(v_in))
        self.Sh.incident(tf.zeros_like(v_in))

        for i in range(sequence_length):
            self.VinRin.set_voltage(input[:, i, 0:1])

            a_c = self.Pc.reflected()
            a_d = self.Pd.reflected()
            a_e = self.B1.reflected()
            a_f = self.R1.reflected()
            a_g = self.Rcollector.reflected()
            a_h = self.Sh.reflected()

            # model_in = tf.concat((a_c, a_d, a_e, a_f, a_g, a_h, Rc, Rd, Re, Rf, Rg, Rh), axis=1)
            model_in = tf.concat((a_c, a_d, a_e, a_f, a_g, a_h), axis=1)
            self.model.incident(tf.transpose(model_in, perm=[0, 2, 1]))

            b_waves = self.model.reflected()
            self.Pc.incident(b_waves[:, :, 0:1])
            self.Pd.incident(b_waves[:, :, 1:2])
            self.B1.incident(b_waves[:, :, 2:3])
            self.R1.incident(b_waves[:, :, 3:4])
            self.Rcollector.incident(b_waves[:, :, 4:5])
            self.Sh.incident(b_waves[:, :, 5:6])

            output = wdf.voltage(self.RL)
            output_sequence = output_sequence.write(i, output)
        
        output_sequence = output_sequence.stack()
        return output_sequence

model = CommonEmitterModel()

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

mse_loss = tf.keras.losses.MeanSquaredError()
loss_func = lambda target, pred: mse_loss(target, pred) + esr_loss(target, pred)

optimizer = tf.keras.optimizers.Adam(learning_rate=4.0)

# %%
def plot_target_pred(target, predicted, epoch):
    plt.figure()
    plt.plot(target[:batch_size], label='Target')
    plt.plot(predicted[:batch_size], '--', label='Predicted')
    plt.xlabel('Time [samples]')
    plt.ylabel('Voltage')
    # plt.show()
    # plt.title(f'Diode Clipper ({diode_name}, {n_layers}x{layer_size}), Epoch {epoch}')
    # plt.legend(loc='lower left')
    
# %%
skip_samples = 0 # skip the first few samples to let state build up
history = { 'loss': [], 'mse': [], 'esr': [] }

# %%
# plot_batch = 2
for epoch in tqdm(range(10000)):
    with tf.GradientTape() as tape:
        outs = tf.transpose(model.forward(data_in_batched)[...,0], perm=[1, 0, 2])
        loss = loss_func(outs[:, skip_samples:, :], data_target_batched[:, skip_samples:, :])

    if epoch % 5 == 0:
        print(f'\nCheckpoint (Epoch = {epoch}):')
        print(f'    Loss: {loss}')
        # target = data_target_batched[plot_batch, skip_samples:, 0]
        # pred = outs[plot_batch, skip_samples:, 0]
        # plot_target_pred(target, pred, epoch)

    history['loss'].append(loss)
    history['mse'].append(mse_loss(outs[:, skip_samples:, :], data_target_batched[:, skip_samples:, :]))
    history['esr'].append(esr_loss(outs[:, skip_samples:, :], data_target_batched[:, skip_samples:, :]))
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

print(f'\nFinal Results:')
print(f'    Loss: {loss}')

# %%
