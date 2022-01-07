# %%
import sys
sys.path.insert(0, '../lib')

import numpy as np
from scipy.special import wrightomega
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import namedtuple
from model_utils import save_model

plots_dir = 'plots/pretraining'

# %%
n_layers = 4
layer_size = 8

DiodeConfig = namedtuple('DiodeConfig', ['name', 'Is', 'nabla', 'Vt'], defaults=['', 1.0e-9, 1.0, 25.85e-3])

default_diode = DiodeConfig('DefaultDiode')
diode_1n4148 = DiodeConfig('1N4148', Is=4.352e-9, nabla=1.906) # borrowed from: https://github.com/neiser/spice-padiwa-amps/blob/master/1N4148.lib

diode_to_train = diode_1n4148

model_name = f'{diode_to_train.name}_{n_layers}x{layer_size}_pretrained'

# %%
# WDF diode pair equations:
# See Werner et al., "An Improved and Generalized Diode Clipper Model for Wave Digital Filters"
# https://www.researchgate.net/publication/299514713_An_Improved_and_Generalized_Diode_Clipper_Model_for_Wave_Digital_Filters

# see reference eqn (18)
def diode_pair_func_good(x, R, diode):
    a = x
    R_Is = diode.Is * R
    Vt = diode.Vt * diode.nabla
    R_Is_overVt = R_Is / Vt
    logR_Is_overVt = np.log(R_Is_overVt)
    lamb = np.sign(a)
    b = a + 2 * lamb * (R_Is - Vt * wrightomega(logR_Is_overVt + lamb * a / Vt + R_Is_overVt))
    return np.float32(b)

# see reference eqn (39)
def diode_pair_func_best(x, R, diode):
    a = x
    R_Is = diode.Is * R
    Vt = diode.Vt * diode.nabla
    R_Is_overVt = R_Is / Vt
    logR_Is_overVt = np.log(R_Is_overVt)
    
    lamb = np.sign(a)
    lamb_a_over_vt = lamb * a / Vt

    b = a - 2 * Vt * lamb * (wrightomega(logR_Is_overVt + lamb_a_over_vt) - wrightomega(logR_Is_overVt - lamb_a_over_vt))
    return np.float32(b)

# %%
N = 1000
test_x = None
for R_order in np.linspace(1, 9, 20):
    R = 10**R_order
    x_vals = np.linspace(-2.5, 2.5, N)
    in_vals = np.stack([x_vals, np.ones(N) * R]).transpose()

    if test_x is None:
        test_x = in_vals
    else:
        test_x = np.concatenate([test_x, in_vals])

# %%
fig, ax1 = plt.subplots()

ax1.set_xlabel('Time [samples]')
ax1.set_ylabel('Incident Wave [V]')
a_plot, = ax1.plot(test_x[:,0], label='a')

ax2 = ax1.twinx()
ax2.set_ylabel('Port Impedance [log(Ohms)]')
R_plot, = ax2.plot(np.log(test_x[:,1]), color='red', label='log(R)')

ax1.legend(handles=[a_plot, R_plot])

plt.title('Diode Network Synthetic Training Data')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(f'{plots_dir}/diodes_synth_training_data.png')

# %%
ideal_y = np.zeros_like(test_x[:,0])
ideal_y2 = np.zeros_like(test_x[:,0])
for n in range(len(ideal_y)):
    # multiply by -1 to make the data line up better
    ideal_y[n] = -1 * diode_pair_func_best(*test_x[n], diode_to_train)

# use log of impedance instead of impedance!
test_x[:,1] = np.log(test_x[:,1])

# %%
plt.plot(test_x[:,0])
plt.plot(ideal_y)

# %%
layer_xs = []
inputs = tf.keras.Input(shape=(2,))
for n in range(n_layers + 2):
    layer_in = inputs if n < 1 else layer_xs[n-1]
    layer_width = layer_size if n < n_layers + 1 else 1
    layer_act = 'tanh' if n < n_layers + 1 else 'linear'
    layer_xs.append(tf.keras.layers.Dense(layer_width,
                                          activation=layer_act,
                                          kernel_initializer='orthogonal')(layer_in))

diode_model = tf.keras.Model(inputs=inputs, outputs=layer_xs[-1])
diode_model.summary()

# %%
mse_loss = tf.keras.losses.MeanSquaredError()

eps = np.finfo(np.float32).eps
def esr_loss(target_y, predicted_y, emphasis_func=lambda x : x):
    target_yp = emphasis_func(target_y)
    pred_yp = emphasis_func(predicted_y)
    mse = tf.math.reduce_sum(tf.math.square(target_yp - pred_yp))
    energy = tf.math.reduce_sum(tf.math.square(target_yp))
    
    loss_unnorm = mse / tf.cast(energy + eps, tf.float32)
    return tf.sqrt(loss_unnorm / N)

pre_loss_mse = mse_loss(ideal_y, diode_model(test_x)).numpy()
pre_loss_esr = esr_loss(ideal_y, diode_model(test_x)).numpy()
print(f'Loss before training: {pre_loss_mse}, {pre_loss_esr}')

@tf.autograph.experimental.do_not_convert
def my_loss(target_y, predicted_y):
    return mse_loss(target_y, predicted_y) + esr_loss(target_y, predicted_y)

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
diode_model.compile(optimizer, my_loss)
diode_model.fit(test_x, ideal_y, epochs=2000)

# %%
y_test = diode_model(test_x).numpy().flatten()

pre_loss_mse = mse_loss(ideal_y, y_test).numpy()
pre_loss_esr = esr_loss(ideal_y, tf.cast(y_test, tf.float32)).numpy()
print(f'Loss after training: {pre_loss_mse}, {pre_loss_esr}')

# %%
plt.plot(-ideal_y, label='Target')
plt.plot(-y_test, '--', label='Predicted')

plt.xlim(0, 20000)
plt.grid()

plt.title(f'Pretrained Diode Output ({diode_to_train.name}, {n_layers}x{layer_size})')
plt.xlabel('Time [samples]')
plt.ylabel('Reflected Wave [V]')
plt.legend()

plt.savefig(f'{plots_dir}/{model_name}.png')

# %%
save_model(diode_model, f'models/pretrained/{model_name}_model.json')

# %%
# Training Results:
# 1N4148:
# - 2x4:  MSE = 1.34e-3, ESR = 1.23e-3
# - 2x8:  MSE = 5.51e-5, ESR = 2.49e-4
# - 2x16: MSE = 7.98e-6, ESR = 9.49e-5
# - 4x4:  MSE = 6.38e-4, ESR = 8.48e-4
# - 4x8:  MSE = 4.43e-5, ESR = 2.24e-4
