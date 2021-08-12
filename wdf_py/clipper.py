# %%
import numpy as np
from scipy.special import wrightomega
import matplotlib.pyplot as plt
import tf_wdf as wdf
from tqdm import tqdm

# %%
BASE_DIR='/Users/jachowdhury/Developer/differentiable-wdfs'
# BASE_DIR='/user/j/jatin/Documents/differentiable-wdfs'
x = np.genfromtxt(f"{BASE_DIR}/test_data/clipper_x.csv", dtype=np.float32)
y_ref = np.genfromtxt(f"{BASE_DIR}/test_data/clipper_y.csv", dtype=np.float32)

print(x.shape)
print(y_ref.shape)

FS = 48000
N = 1200
x = x[:N]
y_ref = y_ref[:N]

# %%
def create_root_model():
    n_layers = 3
    layer_size = 8
    layer_xs = []
    inputs = wdf.tf.keras.Input(shape=(2,))
    for n in range(n_layers):
        layer_in = inputs if n < 1 else layer_xs[n-1]
        layer_width = layer_size if n < n_layers - 1 else 1
        layer_act = 'relu' if n < n_layers - 1 else 'linear'
        layer_xs.append(wdf.tf.keras.layers.Dense(layer_width,
                                                  activation=layer_act,
                                                  kernel_initializer='orthogonal')(layer_in))
    return wdf.tf.keras.Model(inputs=inputs, outputs=layer_xs[-1])

# %%
class ClipperModel(wdf.tf.Module):
    def __init__(self):
        super(ClipperModel, self).__init__('Clipper')
        self.Vs = wdf.ResistiveVoltageSource()
        self.R = wdf.Resistor(10.0e3)
        self.P1 = wdf.WDFParallel(self.Vs, self.R)

        self.root_model = create_root_model()
        self.root_model.summary()

    @wdf.tf.function
    def __call__(self, x):
        y = []
        for n in range(x.shape[-1]):
            self.Vs.set_voltage(x[:,n])
            a = self.P1.reflected()

            in_vec = wdf.tf.stack([a, [self.P1.R]], axis=1)
            b = self.root_model(in_vec)
            # b = wdf.tf.numpy_function(diode_pair_func, [in_vec], wdf.tf.float32)

            self.P1.incident(b)
            y.append(self.R.voltage())
        return wdf.tf.stack(y)

model = ClipperModel()
x_in = np.array([x.transpose()])

# %%
loss_func = wdf.tf.keras.losses.MeanSquaredError(reduction=wdf.tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
pre_loss = loss_func(y_ref, model(x_in)).numpy()
print(f'Loss before training: {pre_loss}')

# %%
def train(model, x, y, optimizer):
    with wdf.tf.GradientTape() as tape:
        current_loss = loss_func(y, model(x)) # compute loss

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(current_loss, model.root_model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.root_model.trainable_weights))

def training_loop(model, x, y, num_epochs):
    optimizer = wdf.tf.keras.optimizers.Adam(learning_rate=1e-2)
    for epoch in tqdm(range(num_epochs)):
        train(model, x, y, optimizer)
        current_loss = loss_func(y, model(x))
        print("Epoch %2d: loss=%2.5f" % (epoch, current_loss))

# %%
training_loop(model, x_in, y_ref, 100)
y_test = model(x_in).numpy().flatten()

# %%
plt.plot(x)
plt.plot(y_ref)
plt.plot(y_test, '--')
plt.savefig(f'{BASE_DIR}/final.png')
plt.show()

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

# %%
N = 100
test_x = np.linspace(-5, 5, N)
test_x = np.stack([test_x, np.ones(N) * 1.0e-9]).transpose()

ideal_y = np.zeros(N)
for n in range(N):
    ideal_y[n] = diode_pair_func(np.array([test_x[n,:]]))

model_y = model.root_model(test_x).numpy().flatten()

plt.plot(test_x[:,0], ideal_y)
plt.plot(test_x[:,0], model_y, '--')

# %%
