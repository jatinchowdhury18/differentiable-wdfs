# %%
import numpy as np
from scipy.special import wrightomega
import matplotlib.pyplot as plt
import tf_wdf as wdf
from tf_wdf import tf
from tqdm import tqdm
import pathlib

from model_utils import *

# %%
BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
x = np.genfromtxt(f"{BASE_DIR}/test_data/clipper_pot_x.csv", dtype=np.float32)
R_data = np.genfromtxt(f"{BASE_DIR}/test_data/clipper_pot_r.csv", dtype=np.float32)
y_ref = np.genfromtxt(f"{BASE_DIR}/test_data/clipper_pot_y.csv", dtype=np.float32)

print(x.shape)
print(y_ref.shape)

FS = 48000
N = len(x)
# x = x[:N]
# y_ref = y_ref[:N]

# %%
n_batches = 1

data_in = np.stack([x, R_data], axis=0).transpose()
data_in_batched = np.array(np.array_split(data_in, n_batches))
data_target = np.transpose(np.array([y_ref]))

print(data_in.shape)
print(data_in_batched.shape)
# print(data_target.shape)

# plt.plot(data_in[:, 0])
plt.plot(data_in_batched[0, :, 1] / 200000)
plt.plot(data_in_batched[0, :, 0] / 8)
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
    def __init__(self, in_size, n_layers, hidden_dim, batch_size=1):
        super(RootModel, self).__init__()

        self.a = tf.Variable(initial_value=tf.zeros(1), name='incident_wave')
        self.b = tf.Variable(initial_value=tf.zeros(1), name='reflected_wave')

        self.layers = []

        for n in range(n_layers):
            in_size = in_size if n == 0 else hidden_dim
            out_size = 1 if n == n_layers - 1 else hidden_dim
            self.layers.append(DenseLayer(batch_size, in_size, out_size))

            if n < n_layers - 1:
                self.layers.append(tf.nn.tanh)

    def incident(self, x):
        self.a = x[:, :, 0]
        self.model_in = x

    def reflected(self):
        x = self.layers[0](self.model_in)
        for l in self.layers[1:]:
            x = l(x)

        self.b = x - self.a
        return self.b

# %%
class ClipperModel(tf.Module):
    def __init__(self):
        super(ClipperModel, self).__init__()
        self.Vs = wdf.ResistiveVoltageSource()
        self.R = wdf.Resistor(10.0e3)
        self.P1 = wdf.Parallel(self.Vs, self.R)

        self.model = RootModel(2, 8, 8, batch_size=n_batches)

    def forward(self, input):
        sequence_length = input.shape[1]
        batch_size = input.shape[0]
        input = tf.cast(tf.expand_dims(input, axis=-1), dtype=tf.float32)
        output_sequence = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False)

        for i in range(sequence_length):
            self.Vs.set_voltage(input[:, i, 0:1])

            self.Vs.set_resistance(input[:, i, 1:2])
            self.P1.calc_impedance()

            model_in = tf.concat((self.P1.reflected(), self.P1.R / 1000), axis=1)
            self.model.incident(tf.transpose(model_in, perm=[0, 2, 1]))
            self.P1.incident(self.model.reflected())

            output = wdf.voltage(self.model)
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
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-9)

# %%
for epoch in tqdm(range(100)):
    with tf.GradientTape() as tape:
        outs = model.forward(data_in_batched)[...,0]
        loss = loss_func(outs, data_target)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 10 == 0:
        print(f'\nCheckpoint (Epoch = {epoch}):')
        print(f'    Loss: {loss}')
        plt.figure()
        plt.plot(data_target[:,0])
        plt.plot(outs.numpy().flatten(), '--')
        plt.show()

print(f'\nFinal Results:')
print(f'    Loss: {loss}')

# %%
outs = model.forward(data_in_batched)[...,0].numpy().flatten()
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

def save_model_json(model):
    def get_weights(layer):
        weights = layer.kernel.numpy()
        bias = layer.bias.numpy()
        return [weights, bias]

    model_dict = {}
    model_dict["in_shape"] = 1
    layers = []
    for layer in model.layers:
        if layer ==  tf.nn.tanh:
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

save_model(model.model, 'clipper.json')

# %%
