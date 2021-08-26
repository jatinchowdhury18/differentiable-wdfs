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

        # self.layers.append(MyLayer)
        # self.layers.append(MyLayer)

        for n in range(n_layers):
            in_size = 1 if n == 0 else hidden_dim
            out_size = 1 if n == n_layers - 1 else hidden_dim
            self.layers.append(DenseLayer(batch_size, in_size, out_size))

            if n < n_layers - 1:
                self.layers.append(tf.nn.relu)

    def forward(self, input):
        for l in self.layers:
            input = l(input)

        return input

# %%
class ClipperModel(tf.Module):
    def __init__(self):
        super(ClipperModel, self).__init__()
        self.Vs = wdf.ResistiveVoltageSource()
        self.R = wdf.Resistor(10.0e3)
        self.P1 = wdf.Parallel(self.Vs, self.R)

        self.model = RootModel(10, 16)

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

model = ClipperModel()

# %%
batch_size = N
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
loss_func = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

# for epoch in tqdm(range(250)):
for epoch in tqdm(range(100)):
    with tf.GradientTape() as tape:
        outs = model.forward(data_in)[...,0]
        loss = loss_func(outs, data_target)
    grads = tape.gradient(loss, model.trainable_variables)

    if epoch % 5 == 0:
        print(f'\nCheckpoint (Epoch = {epoch}):')
        print(f'    Loss: {loss}')
        # print(f'    Grads: {[g.numpy() for g in grads]}')
        # print(f'    Trainables: {[t.numpy() for t in model.trainable_variables]}')
    
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

print(f'\nFinal Results:')
print(f'    Loss: {loss}')
# print(f'    Grads: {[g.numpy() for g in grads]}')
# print(f'    Trainables: {[t.numpy() for t in model.trainable_variables]}')

# %%
outs = model.forward(data_in)[...,0].numpy().flatten()
print(outs.shape)
plt.plot(data_target[:,0])
plt.plot(outs, '--')

# %%
