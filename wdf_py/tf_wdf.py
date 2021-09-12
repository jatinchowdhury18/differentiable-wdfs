import tensorflow as tf
tf.get_logger().setLevel('WARN')

def voltage(wdf):
    return (wdf.a + wdf.b) * tf.constant(0.5)

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

class ResistiveVoltageSource(tf.Module):
    def __init__(self, initial_R = 1.0e-9, trainable = False):
        super(ResistiveVoltageSource, self).__init__()
        self.a = tf.Variable(initial_value=tf.zeros(1), name='incident_wave')
        self.b = tf.Variable(initial_value=tf.zeros(1), name='reflected_wave')

        self.R = tf.Variable(initial_value=initial_R, name='resistance', trainable=trainable)

    def calc_impedance(self):
        pass

    def set_voltage(self, voltage):
        self.Vs = voltage
    
    def set_resistance(self, resistance):
        self.R = resistance

    def incident(self, x):
        self.a = x

    def reflected(self):
        self.b = self.Vs
        return self.b

class Resistor(tf.Module):
    def __init__(self, initial_R, trainable = False):
        super(Resistor, self).__init__()
        self.a = tf.Variable(initial_value=tf.zeros(1), name='incident_wave')
        self.b = tf.Variable(initial_value=tf.zeros(1), name='reflected_wave')

        self.R = tf.Variable(initial_value=initial_R, name='resistance', dtype=tf.float32,
                             trainable=trainable, constraint=lambda z: tf.clip_by_value(z, 180.0, 1.0e6))

    def calc_impedance(self):
        pass

    def incident(self, x):
        self.a = x

    def reflected(self):
        self.b = tf.zeros_like(self.b)
        return self.b

class Capacitor(tf.Module):
    def __init__(self, initial_C, FS, trainable = False):
        super(Capacitor, self).__init__()
        self.a = tf.Variable(initial_value=tf.zeros(1), name='incident_wave')
        self.b = tf.Variable(initial_value=tf.zeros(1), name='reflected_wave')

        self.FS = FS
        self.C = tf.Variable(initial_value=initial_C, name='capacitance', dtype=tf.float32,
                             trainable=trainable, constraint=lambda z: tf.clip_by_value(z, 0.1e-12, 1.0))
        self.R = tf.Variable(initial_value=1.0 / (2.0 * initial_C * FS), name='impedance', trainable=False)
        
        self.z = tf.Variable(initial_value=0.0, name='state', trainable=False)

    def calc_impedance(self):
        self.R = tf.math.reciprocal(self.C * (2.0 * self.FS))

    def incident(self, x):
        self.a = x
        self.z = self.a

    def reflected(self):
        self.b = self.z
        return self.b

class Series(tf.Module):
    def __init__(self, P1, P2):
        super(Series, self).__init__()
        self.a = tf.Variable(initial_value=tf.zeros(1), name='incident_wave')
        self.b = tf.Variable(initial_value=tf.zeros(1), name='reflected_wave')

        self.P1 = P1
        self.P2 = P2

    def calc_impedance(self):
        self.P1.calc_impedance()
        self.P2.calc_impedance()

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

class Parallel(tf.Module):
    def __init__(self, P1, P2):
        super(Parallel, self).__init__()
        self.a = tf.Variable(initial_value=tf.zeros(1), name='incident_wave')
        self.b = tf.Variable(initial_value=tf.zeros(1), name='reflected_wave')

        self.P1 = P1
        self.P2 = P2

    def calc_impedance(self):
        self.P1.calc_impedance()
        self.P2.calc_impedance()

        G1 = 1.0 / self.P1.R
        G2 = 1.0 / self.P2.R
        G = G1 + G2

        self.R = 1.0 / G
        self.p1R = G1 / G

    def incident(self, x):
        b2 = x + self.b_temp
        self.P1.incident(self.b_diff + b2)
        self.P2.incident(b2)
        self.a = x
        
    def reflected(self):
        b1 = self.P1.reflected()
        b2 = self.P2.reflected()

        self.b_diff = b2 - b1
        self.b_temp = -self.p1R * self.b_diff
        self.b = b2 + self.b_temp
        return self.b

class Inverter(tf.Module):
    def __init__(self, P1):
        super(Inverter, self).__init__()
        self.a = tf.Variable(initial_value=tf.zeros(1), name='incident_wave')
        self.b = tf.Variable(initial_value=tf.zeros(1), name='reflected_wave')

        self.P1 = P1

    def calc_impedance(self):
        self.P1.calc_impedance()
        self.R = self.P1.R

    def incident(self, x):
        self.P1.incident(-x)
        self.a = x
        
    def reflected(self):
        self.b = -self.P1.reflected()
        return self.b

