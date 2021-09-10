# %%
import numpy as np
from scipy.special import wrightomega
import matplotlib.pyplot as plt
import tf_wdf as wdf
from tf_wdf import tf
from tqdm import tqdm
import pathlib

# %%
BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
x = np.genfromtxt(f"{BASE_DIR}/test_data/clipper_x.csv", dtype=np.float32)
y_ref = np.genfromtxt(f"{BASE_DIR}/test_data/clipper_y.csv", dtype=np.float32)

print(x.shape)
print(y_ref.shape)

FS = 48000
N = 1200
x = x[:N]
y_ref = y_ref[:N]

# %%
n_batches = 1
FS = 48000
freq = 100

data_in = np.array([x])
data_in_batched = np.array(np.array_split(data_in[0], n_batches))
data_target = np.transpose(np.array([y_ref]))

print(data_in.shape)
print(data_in_batched.shape)
print(data_target.shape)

plt.plot(data_in[0])
plt.plot(data_in_batched[0])
plt.plot(data_target[:,0])

# %%
class DenseLayer(tf.Module):
    """ Dense layer without weights sharing"""
    def __init__(self, batch_size, in_size, out_size):
        super(DenseLayer, self).__init__()
        bs = batch_size
        self.kernel = tf.Variable(self.init_weights(bs, in_size, out_size), dtype=tf.float32)
        self.bias = tf.Variable(self.init_bias(bs, out_size), dtype=tf.float32)

    def init_weights(self, batch_size, size1, size2):
        initializer = tf.keras.initializers.Orthogonal()
        init = initializer(shape=(size1, size2))
        return [init for b in range(batch_size)]

    def init_bias(self, batch_size, size):
        initializer = tf.keras.initializers.Zeros()
        init = initializer(shape=(size))
        return [init for b in range(batch_size)]

    def __call__(self, input):
        return tf.matmul(input, self.kernel) + self.bias

class RootModel(tf.Module):
    def __init__(self, n_layers, hidden_dim, activation='relu', batch_size=1):
        super(RootModel, self).__init__()
        self.layers = []

        for n in range(n_layers):
            in_size = 1 if n == 0 else hidden_dim
            out_size = 1 if n == n_layers - 1 else hidden_dim
            self.layers.append(DenseLayer(batch_size, in_size, out_size))

            if n < n_layers - 1:
                self.layers.append(tf.nn.tanh)

    def forward(self, input):
        x = self.layers[0](input)
        for l in self.layers[1:]:
            x = l(x)

        return x - input

# %%
class ClipperModel(tf.Module):
    def __init__(self):
        super(ClipperModel, self).__init__()
        self.Vs = wdf.ResistiveVoltageSource()
        self.R = wdf.Resistor(10.0e3)
        self.P1 = wdf.Parallel(self.Vs, self.R)

        self.model = RootModel(4, 8, batch_size=n_batches)

    def forward(self, input):
        sequence_length = input.shape[1]
        batch_size = input.shape[0]
        input = tf.cast(tf.expand_dims(input, axis=-1), dtype=tf.float32)
        output_sequence = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False)

        self.P1.calc_impedance()
        for i in range(sequence_length):
            self.Vs.set_voltage(input[:, i])

            a = self.P1.reflected()
            b = self.model.forward(a)
            self.P1.incident(b)

            output = wdf.voltage(self.R)
            output_sequence = output_sequence.write(i, output)
        
        output_sequence = output_sequence.stack()
        return output_sequence

# %%
model = ClipperModel()

def pre_emphasis_filter(x, coeff=0.85):
  return tf.concat([x[0:1], x[1:] - coeff*x[:-1]], axis=0)

eps = np.finfo(float).eps
def esr_loss(target_y, predicted_y, emphasis_func=lambda x : x):
    target_yp = emphasis_func(target_y)
    pred_yp = emphasis_func(predicted_y)
    mse = tf.math.reduce_sum(tf.math.square(target_yp - pred_yp))
    energy = tf.math.reduce_sum(tf.math.square(target_yp))
    
    loss_unnorm = mse / (energy + eps)
    # loss_unnorm = mse * energy
    return loss_unnorm / N

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
# loss_func = esr_loss
loss_func = lambda target, pred: avg_loss(target, pred) \
    + bounds_loss(target, pred) \
    + esr_loss(target, pred)
# loss_func = lambda target, pred: avg_loss(target, pred) \
#     + bounds_loss(target, pred) \
#     + esr_with_emph(target, pred)
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-9)

# %%
for epoch in tqdm(range(500)):
    with tf.GradientTape() as tape:
        outs = model.forward(data_in)[...,0]
        loss = loss_func(outs, data_target)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 50 == 0:
        print(f'\nCheckpoint (Epoch = {epoch}):')
        print(f'    Loss: {loss}')
        plt.figure()
        plt.plot(data_target[:,0])
        plt.plot(outs.numpy().flatten(), '--')
        plt.show()


print(f'\nFinal Results:')
print(f'    Loss: {loss}')

outs = model.forward(data_in)[...,0].numpy().flatten()
print(outs.shape)
plt.plot(data_target[:,0])
plt.plot(outs, '--')

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

N = 5000
test_x = np.linspace(-10, 10, N)
test_x = np.stack([test_x, np.ones(N) * 1.0e-9]).transpose()

ideal_y = np.zeros(N)
for n in range(N):
    ideal_y[n] = diode_pair_func(np.array([test_x[n,:]]))

x_in_tt = np.array([test_x[:,0].astype(np.float32)]).transpose()
y_test = model.model.forward(x_in_tt).numpy().flatten()

plt.plot(test_x[:,0], ideal_y)
plt.plot(test_x[:,0], y_test, '--')

# %%
