# %%
import numpy as np
import tensorflow as tf
import tqdm as tqdm

# %%
# based (pretty loosely) on this differentiable IIR script:
# https://github.com/andreofner/APC/blob/7e3fc13afdef204465670c7e9b9589021ad89755/IIR.py
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
        self.Vs.set_voltage(input)
        self.S1.calc_impedance()

        self.Vs.incident(self.S1.reflected())
        self.S1.incident(self.Vs.reflected())

        return (self.R1.a + self.R1.b) * tf.constant(0.5)

model = Model()

# %%
loss_func = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=5.0)

data_in = np.array([10.0])
data_target = np.array([-5.0])

for epoch in tqdm.tqdm(range(1000)):
    with tf.GradientTape() as tape:
        loss = loss_func(model.forward(data_in), data_target)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# y_test = model.forward(data_in)
# print(f'Y = {y_test.numpy()}, L = {np.sum(loss.numpy())}, R = {model.trainable_variables[0].numpy()}')
print(f'L = {np.sum(loss.numpy())}, R = {model.trainable_variables[0].numpy()}')

# %%
