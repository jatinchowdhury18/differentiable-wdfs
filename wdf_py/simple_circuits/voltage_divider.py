# %%
import sys
sys.path.insert(0, '../lib')

import numpy as np
import tf_wdf as wdf
from tf_wdf import tf
import tqdm as tqdm
import matplotlib.pyplot as plt

# %%
# based loosely on: https://github.com/andreofner/APC/blob/master/IIR.py
class Model(tf.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Vs = wdf.IdealVoltageSource()
        self.R1 = wdf.Resistor(2.0e3, True)
        self.R2 = wdf.Resistor(100.0, True)

        self.S1 = wdf.Series(self.R1, self.R2)
        self.I1 = wdf.Inverter(self.S1)

    def forward(self, input):
        sequence_length = input.shape[1]
        batch_size = input.shape[0]
        input = tf.cast(tf.expand_dims(input, axis=-1), dtype=tf.float32)
        output_sequence = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False)

        self.I1.calc_impedance()
        for i in range(sequence_length):
            self.Vs.set_voltage(input[:, i])

            self.Vs.incident(self.I1.reflected())
            self.I1.incident(self.Vs.reflected())

            output = wdf.voltage(self.R1)
            output_sequence = output_sequence.write(i, output)
        
        output_sequence = output_sequence.stack()
        return output_sequence

# %%
batch_size = 128
n_batches = 4
FS = 48000
freq = 100

data_in = np.array([np.sin(2 * np.pi * np.arange(batch_size * n_batches) * freq / FS)])
data_in_batched = np.array(np.array_split(data_in[0], n_batches))
data_target = np.transpose(data_in * 0.5)

print(data_in.shape)
print(data_in_batched.shape)
print(data_target.shape)

plt.plot(data_in[0])
plt.plot(data_in_batched[0])
plt.plot(data_target[:,0])

# %%
model = Model()
loss_func = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=25.0)

R1s = []
R2s = []
losses = []

for epoch in tqdm.tqdm(range(100)):
    with tf.GradientTape() as tape:
        outs = model.forward(data_in)[...,0]
        loss = loss_func(outs, data_target)
    grads = tape.gradient(loss, model.trainable_variables)

    if epoch % 50 == 0:
        print(f'\nCheckpoint (Epoch = {epoch}):')
        print(f'    Loss: {loss}')
        print(f'    Grads: {[g.numpy() for g in grads]}')
        print(f'    Trainables: {[t.numpy() for t in model.trainable_variables]}')
    
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    R1s.append(model.trainable_variables[0].numpy())
    R2s.append(model.trainable_variables[1].numpy())
    losses.append(loss)

print(f'\nFinal Results:')
print(f'    Loss: {loss}')
print(f'    Grads: {[g.numpy() for g in grads]}')
print(f'    Trainables: {[t.numpy() for t in model.trainable_variables]}')
# %%
outs = model.forward(data_in)[...,0]
plt.plot(data_target[:,0])
plt.plot(outs, '--')

# %%
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Resistance [Ohms]')
R1s_plot, = ax1.plot(R1s, label='R1')
R2s_plot, = ax1.plot(R2s, label='R2')
ax1.legend(loc='upper center')

ax2 = ax1.twinx()
ax2.set_ylabel('Error')
losses_plot, = ax2.plot(losses, color='red', label='Error')

ax1.legend(handles=[R1s_plot, R2s_plot, losses_plot])

plt.title('Diff. Voltage Divider Training')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('plots/voltage_divider.png')

# %%
