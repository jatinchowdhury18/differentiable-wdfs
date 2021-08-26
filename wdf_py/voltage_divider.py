# %%
import numpy as np
import tensorflow as tf
import tqdm as tqdm
import matplotlib.pyplot as plt

# %%
# based (pretty loosely) on this differentiable IIR script:
# https://github.com/andreofner/APC/blob/master/IIR.py
class IdealVoltageSource(tf.Module):
    def __init__(self):
        super(IdealVoltageSource, self).__init__()
        self.a = tf.Variable(initial_value=tf.zeros(1), name='incident_wave')
        self.b = tf.Variable(initial_value=tf.zeros(1), name='reflected_wave')

    def set_voltage(self, voltage):
        self.Vs = voltage

    def incident(self, x):
        self.a = x

    def reflected(self):
        self.b = -self.a + tf.constant(2.0) * self.Vs
        return self.b

class Resistor(tf.Module):
    def __init__(self, initial_R, trainable = False):
        super(Resistor, self).__init__()
        self.a = tf.Variable(initial_value=tf.zeros(1), name='incident_wave')
        self.b = tf.Variable(initial_value=tf.zeros(1), name='reflected_wave')

        self.R = tf.Variable(initial_value=initial_R, name='resistance', trainable=trainable)

    def incident(self, x):
        self.a = x

    def reflected(self):
        self.b = tf.zeros_like(self.b)
        return self.b

class Series(tf.Module):
    def __init__(self, P1, P2):
        super(Series, self).__init__()
        self.a = tf.Variable(initial_value=tf.zeros(1), name='incident_wave')
        self.b = tf.Variable(initial_value=tf.zeros(1), name='reflected_wave')

        self.P1 = P1
        self.P2 = P2

    def calc_impedance(self):
        self.R = self.P1.R + self.P2.R
        self.p1R = self.P1.R / self.R
        self.p2R = self.P2.R / self.R

    def incident(self, x):
        b1 = self.P1.b - self.p1R * (x + self.P1.b + self.P2.b)
        self.P1.incident(b1)
        self.P2.incident(-(x + b1))
        self.a = x
        
    def reflected(self):
        self.b = -(self.P1.reflected() + self.P2.reflected())
        return self.b


# %%
class Model(tf.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Vs = IdealVoltageSource()
        self.R1 = Resistor(1.0e3)
        self.R2 = Resistor(100.0, True)
        self.S1 = Series(self.R1, self.R2)

    def forward(self, input):
        sequence_length = input.shape[1]
        batch_size = input.shape[0]
        input = tf.cast(tf.expand_dims(input, axis=-1), dtype=tf.float32)
        output_sequence = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False)

        self.S1.calc_impedance()
        for i in range(sequence_length):
            self.Vs.set_voltage(input[:, i])

            self.Vs.incident(self.S1.reflected())
            self.S1.incident(self.Vs.reflected())

            output =  (self.R1.a + self.R1.b) * tf.constant(0.5)
            output_sequence = output_sequence.write(i, output)
        output_sequence = output_sequence.stack()
        return output_sequence

model = Model()

# %%
batch_size = 128
n_batches = 5
FS = 48000
freq = 100

data_in = np.array([np.sin(2 * np.pi * np.arange(batch_size * n_batches) * freq / FS)])
data_in_batched = np.array(np.array_split(data_in[0], n_batches))
data_target = np.transpose(data_in * -0.5)

print(data_in.shape)
print(data_in_batched.shape)
print(data_target.shape)

plt.plot(data_in[0])
plt.plot(data_in_batched[0])
plt.plot(data_target[:,0])

# %%
loss_func = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=25.0)

for epoch in tqdm.tqdm(range(250)):
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

print(f'\nFinal Results:')
print(f'    Loss: {loss}')
print(f'    Grads: {[g.numpy() for g in grads]}')
print(f'    Trainables: {[t.numpy() for t in model.trainable_variables]}')
# %%
outs = model.forward(data_in)[...,0]
plt.plot(data_target[:,0])
plt.plot(outs, '--')

# %%
