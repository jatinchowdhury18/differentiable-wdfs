# %%
import numpy as np
import tf_wdf as wdf
from tf_wdf import tf
import tqdm as tqdm
import matplotlib.pyplot as plt
import audio_dspy as adsp
import scipy.signal as signal

FS = 48000

# %%
# based loosely on: https://github.com/andreofner/APC/blob/master/IIR.py
class Model(tf.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Vs = wdf.IdealVoltageSource()
        self.R1 = wdf.Resistor(1000, True)
        self.C1 = wdf.Capacitor(1.0e-6, FS, True)

        self.S1 = wdf.Series(self.R1, self.C1)
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

            output = wdf.voltage(self.C1)
            output_sequence = output_sequence.write(i, output)
        
        output_sequence = output_sequence.stack()
        return output_sequence

model = Model()

# %%
batch_size = 256
n_batches = 5
freq = 720

sweep = adsp.sweep_log(100, 10000, (batch_size * n_batches) / FS, FS)[:batch_size * n_batches]
b, a = adsp.design_LPF1(720, FS)
sweep_filt = signal.lfilter(b, a, sweep)
data_in = np.array([sweep])
data_in_batched = np.array(np.array_split(data_in[0], n_batches))
data_target = np.transpose(np.array([sweep_filt]))

print(data_in.shape)
print(data_in_batched.shape)
print(data_target.shape)

plt.plot(data_in[0])
plt.plot(data_in_batched[0])
plt.plot(data_target[:,0])

# %%
loss_func = tf.keras.losses.MeanSquaredError()
R_optimizer = tf.keras.optimizers.Adam(learning_rate=25.0)
C_optimizer = tf.keras.optimizers.Adam(learning_rate=5.0e-11)

# for epoch in tqdm.tqdm(range(250)):
for epoch in tqdm.tqdm(range(100)):
    with tf.GradientTape() as tape:
        outs = model.forward(data_in)[...,0]
        loss = loss_func(outs, data_target)
    grads = tape.gradient(loss, model.trainable_variables)

    if epoch % 25 == 0:
        print(f'\nCheckpoint (Epoch = {epoch}):')
        print(f'    Loss: {loss}')
        print(f'    Grads: {[g.numpy() for g in grads]}')
        print(f'    Trainables: {[t.numpy() for t in model.trainable_variables]}')
    
    R_optimizer.apply_gradients([(grads[1], model.R1.R)])
    C_optimizer.apply_gradients([(grads[0], model.C1.C)])

print(f'\nFinal Results:')
print(f'    Loss: {loss}')
print(f'    Grads: {[g.numpy() for g in grads]}')
print(f'    Trainables: {[t.numpy() for t in model.trainable_variables]}')
# %%
final_freq = 1.0 / (2 * np.pi * model.R1.R * model.C1.C)
print(final_freq)

outs = model.forward(data_in)[...,0]
plt.plot(data_target[:,0])
plt.plot(outs, '--')

# %%
