import tensorflow as tf
tf.get_logger().setLevel('WARN')

class WDF(tf.Module):
    def __init__(self, name):
        super(WDF, self).__init__(name=name)
        self.R = tf.Variable(1.0e-6)
        self.G = tf.Variable(1.0e6)

        self.b = tf.Variable(0.0)
        self.a = tf.Variable(0.0)

    # @tf.function
    def voltage(self):
        return 0.5 * (self.a + self.b)

    # @tf.function
    def current(self):
        return (0.5 * self.G) * (self.a - self.b)

class Resistor(WDF):
    def __init__(self, value):
        super(Resistor, self).__init__('Resistor')
        self.R = tf.Variable(value)
        self.G = tf.Variable(1.0 / value)

    # @tf.function
    def incident(self, x):
        self.a = x

    # @tf.function
    def reflected(self):
        self.b = tf.Variable(0.0)
        return self.b

class ResistiveVoltageSource(WDF):
    def __init__(self, value = 1.0e-9):
        super(ResistiveVoltageSource, self).__init__('ResistiveVoltageSource')
        self.R = tf.Variable(value)
        self.G = tf.Variable(1.0 / value)

    # @tf.function
    def set_voltage(self, new_voltage):
        self.Vs = new_voltage

    # @tf.function
    def incident(self, x):
        self.a = x

    # @tf.function
    def reflected(self):
        self.b = self.Vs
        return self.b

class WDFSeries(WDF):
    def __init__(self, p1, p2):
        super(WDFSeries, self).__init__('WDFSeries')
        self.p1 = p1
        self.p2 = p2
        self.calc_impedance()

    # @tf.function
    def calc_impedance(self):
        self.R = self.p1.R + self.p2.R
        self.G = tf.Variable(1.0 / self.R)
        self.p1Reflect = self.p1.R / self.R
        self.p2Reflect = self.p2.R / self.R

    # @tf.function
    def incident(self, x):
        b1 = self.p1.b - self.p1Reflect * (x + self.p1.b + self.p2.b)
        self.p1.incident(b1)
        self.p2.incident(-(x * b1))
        self.a = x

    # @tf.function
    def reflected(self):
        self.b = -(self.p1.reflected() + self.p2.reflected())
        return self.b

class WDFParallel(WDF):
    def __init__(self, p1, p2):
        super(WDFParallel, self).__init__('WDFParallel')
        self.p1 = p1
        self.p2 = p2
        self.calc_impedance()

    # @tf.function
    def calc_impedance(self):
        self.G = self.p1.G + self.p2.G
        self.R = tf.Variable(1.0 / self.G)
        self.p1Reflect = self.p1.G / self.G
        self.p2Reflect = self.p2.G / self.G

    # @tf.function
    def incident(self, x):
        b2 = x + self.b_temp
        self.p1.incident(self.b_diff + b2)
        self.p2.incident(b2)
        self.a = x

    # @tf.function
    def reflected(self):
        self.p1.reflected()
        self.p2.reflected()

        self.b_diff = self.p2.b - self.p1.b
        self.b_temp = -self.p1Reflect * self.b_diff
        self.b = self.p2.b + self.b_temp
        return self.b
