# %%
import matplotlib.pyplot as plt
import pickle
import numpy as np

# %%()
name = "Test History"
<<<<<<< Updated upstream
history_file = './histories/1N4148 (1U-2D)_2x16_training_2_history.pkl'
=======
history_file = './histories/1N4148 (1U-1D)_2x16_training_1_hpf_history.pkl'
>>>>>>> Stashed changes

with open(history_file, "rb") as f:
   history = pickle.load(f)

# %%
loss = history['loss']
mse = history['mse']
esr = history['esr']
loss = history['val_loss']
mse = history['val_mse']
esr = history['val_esr']

# %%
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE')
<<<<<<< Updated upstream
mse_plot, = ax1.plot(mse[:500], label='MSE')

ax2 = ax1.twinx()
ax2.set_ylabel('ESR')
esr_plot, = ax2.plot(esr[:500], color='orange', label='ESR')
=======
mse_plot, = ax1.plot(mse[:], label='MSE')

ax2 = ax1.twinx()
ax2.set_ylabel('ESR')
esr_plot, = ax2.plot(esr[:], color='orange', label='ESR')
>>>>>>> Stashed changes

ax1.legend(handles=[mse_plot, esr_plot])

plt.title(f'{name} Training History')

# %%
<<<<<<< Updated upstream
print(history['val_loss'][0])
print(min(history['loss'][:100]))
print(min(history['loss'][:500]))
=======
print(np.format_float_scientific(history['val_loss'][0]))
print(np.format_float_scientific(history['val_loss'][100]))
print(np.format_float_scientific(history['val_loss'][500]))
>>>>>>> Stashed changes

# %%
