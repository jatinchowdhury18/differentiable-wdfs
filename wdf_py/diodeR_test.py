# %%
import numpy as np
from scipy.special import wrightomega
import matplotlib.pyplot as plt
import tensorflow as tf
from model_utils import save_model

# %%
def diode_pair_func(x, R = 1.0e-9):
    a = x
    Vt = 25.85e-3
    R_Is = 1.0e-9 * R
    R_Is_overVt = R_Is / Vt
    logR_Is_overVt = np.log(R_Is_overVt)
    lamb = np.sign(a)
    b = a + 2 * lamb * (R_Is - Vt * wrightomega(logR_Is_overVt + lamb * a / Vt + R_Is_overVt))
    return np.float32(b)

# %%
N = 500
test_x = None
for R_order in range(-9, 10):
    R = 10**R_order
    x_vals = np.linspace(-10, 10, N)
    in_vals = np.stack([x_vals, np.ones(N) * R]).transpose()

    if test_x is None:
        test_x = in_vals
    else:
        test_x = np.concatenate([test_x, in_vals])

plt.plot(test_x[:,0])
plt.plot(np.log(test_x[:,1]) / 2)

# %%
ideal_y = np.zeros_like(test_x[:,0])
for n in range(len(ideal_y)):
    # multiply by -1 to make the data line up better
    ideal_y[n] = -1 * diode_pair_func(*test_x[n])

test_x[:,1] = np.log(test_x[:,1])

# %%
# plt.plot(test_x[:,0] + 5 * np.log(test_x[:,1]), ideal_y)
plt.plot(test_x[:,0])
plt.plot(test_x[:,1] / 2)
plt.plot(ideal_y)

# %%
n_layers = 3
layer_size = 8
layer_xs = []
inputs = tf.keras.Input(shape=(2,))
for n in range(n_layers):
    layer_in = inputs if n < 1 else layer_xs[n-1]
    layer_width = layer_size if n < n_layers - 1 else 1
    layer_act = 'relu' if n < n_layers - 1 else 'linear'
    layer_xs.append(tf.keras.layers.Dense(layer_width,
                                          activation=layer_act,
                                          kernel_initializer='orthogonal')(layer_in))

diode_model = tf.keras.Model(inputs=inputs, outputs=layer_xs[-1])
diode_model.summary()

# %%
loss_func = tf.keras.losses.MeanSquaredError()

pre_loss = loss_func(ideal_y, diode_model(test_x)).numpy()
print(f'Loss before training: {pre_loss}')

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
diode_model.compile(optimizer, loss_func)
diode_model.fit(test_x, ideal_y, epochs=1000)

y_test = diode_model(test_x).numpy().flatten()

# %%
plt.plot(ideal_y)
plt.plot(y_test, '--')

# %%
diode_model.save('diodeR_test_model')

# %%
save_model(diode_model, 'diodeR_test_model.json')

# %%
