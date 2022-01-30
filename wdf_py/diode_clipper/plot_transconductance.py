# %%
import sys

sys.path.insert(0, "../lib")
sys.path.insert(0, "./models")

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from layers import DenseRootModel
import json

# %%
model_file = "./models/1N4148 (1U-1D)_2x16_training_10.json"
with open(model_file, "r") as read_file:
    model_json = json.load(read_file)

model = DenseRootModel(model_json)

# %%
a_waves = np.linspace(-10, 10, num=100, dtype=np.float32)
R_vals = [100, 1000, 10000]
b_waves_out = []

for R in R_vals:
    model_in = np.array([np.stack((a_waves, np.ones_like(a_waves) * np.log(R)))])
    model.incident(np.transpose(model_in, axes=[0, 2, 1]))
    b_waves_out.append(model.reflected().numpy().flatten())

# %%
i_vals_out = []
v_vals_out = []

for i, R in enumerate(R_vals):
    i_vals_out.append((1.0 / (2.0 * R)) * (a_waves - b_waves_out[i]))
    v_vals_out.append((1.0 / 2.0) * (a_waves + b_waves_out[i]))


# %%
Is = 4.352e-9
nabla = 1.906
Vt = 25.85e-3

span = 1.2
v_shockley = np.linspace(-span, span, num=100)
i_shockley = 2 * Is * (np.sinh(v_shockley / (Vt * nabla)))

plt.plot(v_shockley, 1000 * i_shockley)
plt.plot(v_vals_out[0], 1000 * i_vals_out[0], "--")
# for i, R in enumerate(R_vals):
# plt.plot(v_vals_out[i], 1000 * i_vals_out[i], '--')

plt.xlim(-2.5, 2.5)
plt.ylim(-65, 65)
plt.xlabel("Voltage [V]")
plt.ylabel("Current [mA]")

plt.grid()
plt.title("Diode Network Transconductance (1U-1D)")
plt.legend(["Ideal Model", "2x16 Model"])
plt.savefig("./plots/scratch/transconductance.png")

# %%
