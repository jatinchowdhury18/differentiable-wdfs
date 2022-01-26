# %%
import matplotlib.pyplot as plt
import pickle

name = "Test History"

colrow = {
    "col":[0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2],
    "row":[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5],
    }

learning_rates_dict = {
    "learning_rate":[1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3],
    "beta1":[0.9,0.9,0.9,0.7,0.7,0.7,0.5,0.5,0.5,0.9,0.9,0.9,0.7,0.7,0.7,0.5,0.5,0.5],
    "beta2":[0.999,0.8,0.7,0.999,0.8,0.7,0.999,0.8,0.7,0.999,0.8,0.7,0.999,0.8,0.7,0.999,0.8,0.7]
    
}

fig, axs = plt.subplots(6,3)
# fig.suptitle(f'{name} Training History')
fig.set_size_inches(15,12)
fig.tight_layout()
fig.subplots_adjust(wspace=0.4, hspace=0.6)


for train_num in range(18):
    plot_num = train_num + 1
    print(plot_num)
    print([colrow["row"][train_num],colrow["col"][train_num]])
    history_file = f"./histories/1N4148 (2U-2D)_2x8_training_{plot_num}_history.pkl"

    with open(history_file, "rb") as f:
        history = pickle.load(f)

    loss = history['loss']
    mse = history['mse']
    esr = history['esr']

    val_loss = history["val_loss"]
    val_mse = history["val_mse"]
    val_esr = history["val_esr"]
    
    learning_rate = learning_rates_dict["learning_rate"][train_num]
    beta1 = learning_rates_dict["beta1"][train_num]
    beta2 = learning_rates_dict["beta2"][train_num]

    mse_plot, = axs[colrow["row"][train_num],colrow["col"][train_num]].plot(mse, label='MSE')
    axs[colrow["row"][train_num],colrow["col"][train_num]].set_xlabel('Epoch')
    axs[colrow["row"][train_num],colrow["col"][train_num]].set_ylabel('MSE')
       
    ax2 = axs[colrow["row"][train_num],colrow["col"][train_num]].twinx()
    esr_plot, = ax2.plot(esr, color='orange', label='ESR')
    ax2.set_ylabel('ESR')

    axs[colrow["row"][train_num],colrow["col"][train_num]].legend(handles=[mse_plot, esr_plot])
    plot_name = f"Learning Rate: {learning_rate}, beta1 = {beta1}, beta2 = {beta2}"
    axs[colrow["row"][train_num],colrow["col"][train_num]].set_title(plot_name, loc="left")


fig.savefig("./plots/learningrateplot.jpg")
# %%
