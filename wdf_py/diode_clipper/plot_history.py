# %%
import matplotlib.pyplot as plt
import pickle

# %%
name = "Test History"
history_file = './histories/1N4148 (1U-1D)_2x8_training_1_history.pkl'

with open(history_file, "rb") as f:
   history = pickle.load(f)

# %%
loss = history['loss']
mse = history['mse']
esr = history['esr']

# %%
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE')
mse_plot, = ax1.plot(mse[:100], label='MSE')

ax2 = ax1.twinx()
ax2.set_ylabel('ESR')
esr_plot, = ax2.plot(esr[:100], color='orange', label='ESR')

ax1.legend(handles=[mse_plot, esr_plot])

plt.title(f'{name} Training History')

# %%
print(history['val_loss'][0])
print(history['val_loss'][100])
print(history['val_loss'][500])

# %%
