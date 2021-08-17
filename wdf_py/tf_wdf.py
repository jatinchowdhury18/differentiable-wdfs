import tensorflow as tf
tf.get_logger().setLevel('WARN')
from tensorflow.compat.v1 import variable_scope

class WDF(tf.Module):
    def __init__(self, name):
        super(WDF, self).__init__(name=name)
        self.name = name

        with variable_scope(self.name):
            self.R = tf.Variable(1.0e-6)
            self.G = tf.Variable(1.0e6)

            self.b = tf.Variable(0.0) # reflected wave
            self.a = tf.Variable(0.0) # incident wave

    def voltage(self):
        with variable_scope(self.name):
            return 0.5 * (self.a + self.b)

    def current(self):
        with variable_scope(self.name):
            return (0.5 * self.G) * (self.a - self.b)

class Resistor(WDF):
    def __init__(self, name, value):
        super(Resistor, self).__init__(name)
        
        with variable_scope(self.name):
            self.R = tf.Variable(value)
            self.G = tf.Variable(1.0 / value)

    def incident(self, x):
        with variable_scope(self.name):
            self.a = x

    def reflected(self):
        with variable_scope(self.name):
            self.b = tf.zeros_like(self.b)
            return self.b

class ResistiveVoltageSource(WDF):
    def __init__(self, name, value = 1.0e-9):
        super(ResistiveVoltageSource, self).__init__(name)

        with variable_scope(self.name):
            self.R = tf.Variable(value)
            self.G = tf.Variable(1.0 / value)

    def set_voltage(self, new_voltage):
        with variable_scope(self.name):
            self.Vs = tf.identity(new_voltage)

    def incident(self, x):
        with variable_scope(self.name):
            self.a = x

    def reflected(self):
        with variable_scope(self.name):
            self.b = self.Vs
            return self.b

class WDFSeries(WDF):
    def __init__(self, name, p1, p2):
        super(WDFSeries, self).__init__(name)

        with variable_scope(self.name):
            self.p1 = p1
            self.p2 = p2
        self.calc_impedance()

    def calc_impedance(self):
        with variable_scope(self.name):
            self.R = self.p1.R + self.p2.R
            self.G = tf.math.reciprocal(self.R)
            self.p1Reflect = self.p1.R / self.R
            self.p2Reflect = self.p2.R / self.R

    def incident(self, x):
        with variable_scope(self.name):
            b1 = self.p1.b - self.p1Reflect * (x + self.p1.b + self.p2.b)
            self.p1.incident(b1)
            self.p2.incident(-(x * b1))
            self.a = x

    def reflected(self):
        with variable_scope(self.name):
            self.b = -(self.p1.reflected() + self.p2.reflected())
            return self.b

class WDFParallel(WDF):
    def __init__(self, name, p1, p2):
        super(WDFParallel, self).__init__(name)

        with variable_scope(self.name):
            self.p1 = p1
            self.p2 = p2
        self.calc_impedance()

    def calc_impedance(self):
        with variable_scope(self.name):
            self.G = self.p1.G + self.p2.G
            self.R = tf.math.reciprocal(self.G)
            self.p1Reflect = self.p1.G / self.G
            self.p2Reflect = self.p2.G / self.G

    def incident(self, x):
        with variable_scope(self.name):
            b2 = x + self.b_temp
            self.p1.incident(self.b_diff + b2)
            self.p2.incident(b2)
            self.a = x

    def reflected(self):
        b1 = self.p1.reflected()
        b2 = self.p2.reflected()

        with variable_scope(self.name):
            self.b_diff = b2 - b1
            self.b_temp = -self.p1Reflect * self.b_diff
            self.b = b2 + self.b_temp
            return self.b
