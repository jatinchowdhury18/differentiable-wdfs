# %%
import numpy as np
from scipy.special import wrightomega
import matplotlib.pyplot as plt
import tensorflow as tf

# %%
def diode_pair_func(x):
    a = x[0, 0]
    nextR = x[0, 1]
    Vt = 25.85e-3
    R_Is = nextR * 1.0e-9
    R_Is_overVt = R_Is / Vt
    logR_Is_overVt = np.log(R_Is_overVt)
    lamb = np.sign(a)
    b = a + 2 * lamb * (R_Is - Vt * wrightomega(logR_Is_overVt + lamb * a / Vt + R_Is_overVt))
    return np.float32(b)

# %%
N = 5000
test_x = np.linspace(-10, 10, N)
test_x = np.stack([test_x, np.ones(N) * 1.0e-9]).transpose()

ideal_y = np.zeros(N)
for n in range(N):
    ideal_y[n] = diode_pair_func(np.array([test_x[n,:]]))

# %%
plt.plot(test_x[:,0], ideal_y)

# %%
n_layers = 2
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
loss_func = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

pre_loss = loss_func(ideal_y, diode_model(test_x)).numpy()
print(f'Loss before training: {pre_loss}')

# %%
def training_loop(model, x, y, num_epochs, batch_size=32):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    for epoch in range(num_epochs):
        sample_idx = 0
        while sample_idx < len(x):
            in_vec = x[sample_idx : sample_idx + batch_size]
            min_func = lambda : loss_func(y, model(in_vec))
            optimizer.minimize(min_func, [model.trainable_weights])

            sample_idx += batch_size

        current_loss = loss_func(y, model(x))
        print("Epoch %2d: loss=%2.5f" % (epoch, current_loss))

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
diode_model.compile(optimizer, loss_func)
diode_model.fit(test_x, ideal_y, epochs=10)

# training_loop(diode_model, test_x, ideal_y, 10)

y_test = diode_model(test_x).numpy().flatten()
# %%
plt.plot(test_x[:,0], ideal_y)
plt.plot(test_x[:,0], y_test, '--')
# %%
