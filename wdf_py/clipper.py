import numpy as np
import matplotlib.pyplot as plt
import tf_wdf as wdf
from tqdm import tqdm

x = np.genfromtxt("test_data/clipper_x.csv", dtype=np.float32)
y_ref = np.genfromtxt("test_data/clipper_y.csv", dtype=np.float32)

print(x.shape)
print(y_ref.shape)

FS = 48000
N = 1200
x = x[:N]
y_ref = y_ref[:N]

class ClipperModel(wdf.tf.Module):
    def __init__(self):
        super(ClipperModel, self).__init__('Clipper')
        self.Vs = wdf.ResistiveVoltageSource()
        self.R = wdf.Resistor(10.0e3)
        self.P1 = wdf.WDFParallel(self.Vs, self.R)
        
        n_layers = 5
        layer_size = 4
        self.layer_xs = []
        self.inputs = wdf.tf.keras.Input(shape=(1,))
        for n in range(n_layers):
            layer_in = self.inputs if n < 1 else self.layer_xs[n-1]
            layer_width = layer_size if n < n_layers - 1 else 1
            self.layer_xs.append(wdf.tf.keras.layers.Dense(layer_width,
                                                           activation='swish',
                                                           kernel_initializer='random_uniform')(layer_in))
            
        self.root_model = wdf.tf.keras.Model(inputs=self.inputs, outputs=self.layer_xs[-1])

    def __call__(self, x):
        y = []
        for n in range(x.shape[-1]):
            self.Vs.set_voltage(x[:,n])
            a = self.P1.reflected()
            b = self.root_model(a)
            # in_vec = wdf.tf.stack([a, [self.P1.R]], axis=1)
            # b = self.root_model(in_vec)
            self.P1.incident(b)
            y.append(self.R.voltage())
        return wdf.tf.stack(y)

model = ClipperModel()
x_in = np.array([x.transpose()])

loss_func = wdf.tf.keras.losses.MeanSquaredError(reduction=wdf.tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
# def loss_func(target_y, pred_y):
#     mse = wdf.tf.math.reduce_sum(wdf.tf.math.square(target_y - pred_y))
#     energy = wdf.tf.math.reduce_sum(wdf.tf.math.square(target_y))
#     return mse / energy

pre_loss = loss_func(y_ref, model(x_in)).numpy()
print(f'Loss before training: {pre_loss}')

def train(model, x, y, optimizer):
    with wdf.tf.GradientTape() as tape:
        # compute loss
        current_loss = loss_func(y, model(x))

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(current_loss, model.root_model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.root_model.trainable_weights))

def training_loop(model, x, y, num_epochs):
    optimizer = wdf.tf.keras.optimizers.Adam(learning_rate=1e-2)
    for epoch in tqdm(range(num_epochs)):
        # Update the model with the single giant batch
        train(model, x, y, optimizer)

        current_loss = loss_func(y, model(x))
        print("Epoch %2d: loss=%2.5f" % (epoch, current_loss))

training_loop(model, x_in, y_ref, 400)
y_test = model(x_in).numpy().flatten()

plt.plot(x)
plt.plot(y_ref)
plt.plot(y_test, '--')
plt.savefig('final.png')
plt.show()
