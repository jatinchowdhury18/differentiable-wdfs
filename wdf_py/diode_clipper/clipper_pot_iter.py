 # %%
import sys

sys.path.insert(0, "../lib")
sys.path.insert(0, "./models")

import numpy as np
import matplotlib.pyplot as plt

import tf_wdf as wdf
from tf_wdf import tf
from layers import DenseRootModel, DenseLayer

from tqdm import tqdm
from pathlib import Path
import json
import pickle

from diode_config import (
    diode_1n4148_1u1d,
    diode_1n4148_1u2d,
    diode_1n4148_1u3d,
    diode_1n4148_2u2d,
    diode_1n4148_2u3d,
    diode_1n4148_3u3d,
)

from model_utils import *
from dataimport import load_diode_data

BASE_DIR = Path(__file__).parent.parent.parent.resolve()

learning_rates_dict = {
    "learning_rate":[1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3],
    "beta1":[0.9,0.9,0.9,0.7,0.7,0.7,0.5,0.5,0.5,0.9,0.9,0.9,0.7,0.7,0.7,0.5,0.5,0.5],
    "beta2":[0.999,0.8,0.7,0.999,0.8,0.7,0.999,0.8,0.7,0.999,0.8,0.7,0.999,0.8,0.7,0.999,0.8,0.7]
    
}

for learning_rates_num in range(18):
    if learning_rates_num < 17:
        continue

    n_layers = 2
    layer_size = 8
    diode = diode_1n4148_2u2d
    training_number = learning_rates_num+1

    pretrained_model = f"{diode.name}_{n_layers}x{layer_size}_pretrained"
    model_name = f"{diode.name}_{n_layers}x{layer_size}_training_{training_number}"
    plots_dir = Path(f"./plots/{model_name}")

    assert not plots_dir.exists(), "Plots for this training run have already been created!"
    plots_dir.mkdir()

    
    C_val = 4.7e-9
    train_data, train_N, val_data, val_N, FS = load_diode_data(diode, BASE_DIR)

    print(train_data.shape)
    print(val_data.shape)

    
    batch_size = 2048


    def batch_data(data, N):
        x = data[0]
        R_data = data[1]
        y_ref = data[2]
        n_batches = N // batch_size

        data_in = np.stack([x, R_data], axis=0).transpose()
        data_in_trim = data_in[: (n_batches * batch_size), :]
        data_in_batched = np.stack(np.array_split(data_in_trim, n_batches))

        data_target = np.transpose(np.array([y_ref]))
        data_target_trim = data_target[: (n_batches * batch_size), :]
        data_target_batched = np.stack(np.array_split(data_target_trim, n_batches))

        print(data_in.shape)
        print(data_in_batched.shape)
        print(data_target.shape)
        print(data_target_batched.shape)

        return data_in_batched, data_target_batched


    train_X, train_Y = batch_data(train_data, train_N)
    val_X, val_Y = batch_data(val_data, val_N)

    
    plot_batch = 330
    plt.plot(val_X[plot_batch, :, 0])
    plt.plot(val_Y[plot_batch, :, 0])

    
    class ClipperModel(tf.Module):
        def __init__(self, json):
            super(ClipperModel, self).__init__()
            self.Vs = wdf.ResistiveVoltageSource(45.0e3)
            self.C = wdf.Capacitor(C_val, FS)
            self.P1 = wdf.Parallel(self.Vs, self.C)

            self.model = DenseRootModel(json)

        def forward(self, input):
            sequence_length = input.shape[1]
            input = tf.cast(tf.expand_dims(input, axis=-1), dtype=tf.float32)
            output_sequence = tf.TensorArray(
                dtype=tf.float32, size=sequence_length, clear_after_read=False
            )

            self.Vs.reset()
            self.C.reset()

            for i in range(sequence_length):
                self.Vs.set_voltage(input[:, i, 0:1])

                self.Vs.set_resistance(input[:, i, 1:2])
                self.P1.calc_impedance()

                model_in = tf.concat((self.P1.reflected(), tf.math.log(self.P1.R)), axis=1)
                self.model.incident(tf.transpose(model_in, perm=[0, 2, 1]))
                self.P1.incident(self.model.reflected())

                output = wdf.voltage(self.C)
                output_sequence = output_sequence.write(i, output)

            output_sequence = output_sequence.stack()
            return output_sequence


    
    with open(f"./models/pretrained/{pretrained_model}_model.json", "r") as read_file:
        model_json = json.load(read_file)

    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy) 
    model = ClipperModel(model_json)

    
    def pre_emphasis_filter(x, coeff=0.85):
        return tf.concat([x[0:1], x[1:] - coeff * x[:-1]], axis=0)


    eps = np.finfo(float).eps


    def esr_loss(target_y, predicted_y, emphasis_func=lambda x: x):
        target_yp = emphasis_func(target_y)
        pred_yp = emphasis_func(predicted_y)
        mse = tf.math.reduce_sum(tf.math.square(target_yp - pred_yp))
        energy = tf.math.reduce_sum(tf.math.square(target_yp))

        loss_unnorm = mse / tf.cast(energy + eps, tf.float32)
        N = tf.cast((tf.shape(target_y)[0] * tf.shape(target_y)[1]), tf.float32)
        return tf.sqrt(loss_unnorm / N)


    esr_with_emph = lambda target, pred: esr_loss(target, pred, pre_emphasis_filter)


    def avg_loss(target_y, pred_y):
        target_mean = tf.math.reduce_mean(target_y)
        pred_mean = tf.math.reduce_mean(pred_y)
        return tf.math.abs(target_mean - pred_mean)


    def bounds_loss(target_y, pred_y):
        target_min = tf.math.reduce_min(target_y)
        target_max = tf.math.reduce_max(target_y)
        pred_min = tf.math.reduce_min(pred_y)
        pred_max = tf.math.reduce_max(pred_y)
        return tf.math.abs(target_min - pred_min) + tf.math.abs(target_max - pred_max)


    mse_loss = tf.keras.losses.MeanSquaredError()
    loss_func = lambda target, pred: mse_loss(target, pred) + esr_loss(target, pred)


    # optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates_dict["learning_rate"][learning_rates_num], beta_1=learning_rates_dict["beta1"][learning_rates_num], beta_2=learning_rates_dict["beta2"][learning_rates_num])
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # def plot_target_pred(target, predicted, epoch):
    #     plt.figure()
    #     plt.plot(target[:batch_size], label="Target")
    #     plt.plot(predicted[:batch_size], "--", label="Predicted")
        
    #     plt.xlabel("Time [samples]")
    #     plt.ylabel("Voltage")

    #     plt.title(f"Diode Clipper ({diode.name}, {n_layers}x{layer_size}), Epoch {epoch}")
    #     plt.legend(loc="lower left")

    #     plt.savefig(f"./{plots_dir}/epoch_{epoch}.png")
    #     plt.close()

    def plot_target_pred(target, predicted, val_target, val_predicted, epoch):
        fig, axs = plt.subplots(2, 1)
        # plt.figure()
        axs[0].plot(target[:batch_size], label="Target")
        axs[0].plot(predicted[:batch_size], "--", label="Predicted")
        axs[0].set_xlabel("Time [samples]")
        axs[0].set_ylabel("Voltage")
        axs[0].grid(True)
        axs[0].legend(loc="lower left")
        axs[0].set_title("Training",loc="left")


        axs[1].plot(val_target[:batch_size], label="Target")
        axs[1].plot(val_predicted[:batch_size], "--", label="Predicted")
        axs[1].set_xlabel("Time [samples]")
        axs[1].set_ylabel("Voltage")
        axs[1].grid(True)
        axs[1].legend(loc="lower left")
        axs[1].set_title("Validation", loc="left")
        fig.tight_layout()

        fig.suptitle(f"Diode Clipper ({diode.name}, {n_layers}x{layer_size}), Epoch {epoch}")
        fig.set_size_inches(7.5, 5)
        fig.subplots_adjust(wspace=0, hspace=0.4)

        plt.show()

        fig.savefig(f"./{plots_dir}/epoch_{epoch}.png")
        plt.close()

    
    skip_samples = 50  # skip the first few samples to let state build up
    history = {
        "loss": [],
        "mse": [],
        "esr": [],
        "val_loss": [],
        "val_mse": [],
        "val_esr": [],
    }


    
    for epoch in tqdm(range(101)):
        with tf.GradientTape() as tape:
            outs = tf.transpose(model.forward(train_X)[..., 0], perm=[1, 0, 2])
            loss = loss_func(outs[:, skip_samples:, :], train_Y[:, skip_samples:, :])
            scaled_loss = optimizer.get_scaled_loss(loss)

        history["loss"].append(loss)
        history["mse"].append(
            mse_loss(outs[:, skip_samples:, :], train_Y[:, skip_samples:, :])
        )
        history["esr"].append(
            esr_loss(outs[:, skip_samples:, :], train_Y[:, skip_samples:, :])
        )

        val_outs = tf.transpose(model.forward(val_X)[..., 0], perm=[1, 0, 2])
        val_loss = loss_func(val_outs[:, skip_samples:, :], val_Y[:, skip_samples:, :])
        history["val_loss"].append(val_loss)
        history["val_mse"].append(
            mse_loss(val_outs[:, skip_samples:, :], val_Y[:, skip_samples:, :])
        )
        history["val_esr"].append(
            esr_loss(val_outs[:, skip_samples:, :], val_Y[:, skip_samples:, :])
        )

        grads = tape.gradient(scaled_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 5 == 0:
            print(f"\nCheckpoint (Epoch = {epoch}):")
            print(f"    Loss: {loss}")
            print(f"    Val Loss: {val_loss}")
            train_target = train_Y[plot_batch, skip_samples:, 0]
            train_pred = outs[plot_batch, skip_samples:, 0]
            val_target = val_Y[plot_batch, skip_samples:, 0]
            val_pred = val_outs[plot_batch, skip_samples:, 0]
            # plot_target_pred(val_target, val_pred, epoch)
            plot_target_pred(train_target, train_pred, val_target, val_pred, epoch)

    print(f"\nFinal Results:")
    print(f"    Loss: {loss}")
    with open(f"./histories/{model_name}_history.pkl", "wb") as f:
        pickle.dump(history, f)

    outs = tf.transpose(model.forward(train_X)[..., 0], perm=[1, 0, 2])
    
    val_outs = tf.transpose(model.forward(val_X)[..., 0], perm=[1, 0, 2])
    
    train_target = train_Y[plot_batch, skip_samples:, 0]
    train_pred = outs[plot_batch, skip_samples:, 0]
    val_target = val_Y[plot_batch, skip_samples:, 0]
    val_pred = val_outs[plot_batch, skip_samples:, 0]
    plot_target_pred(train_target, train_pred, val_target, val_pred, "final")
    # plt.ylim(-0.2, 0.2)
    # plt.xlim(11150, 11900)

    
    def save_model_json(model):
        def get_weights(layer):
            weights = layer.kernel.numpy()[0]
            bias = layer.bias.numpy()[0]
            return [weights, bias]

        model_dict = {}
        model_dict["in_shape"] = (None, 2)
        layers = []
        for layer in model.layers:
            if layer == tf.nn.tanh:
                layers[-1]["activation"] = "tanh"
                continue

            if isinstance(layer, DenseLayer):
                layer_dict = {
                    "type": "dense",
                    "shape": (None, layer.bias.shape[-1]),
                    "weights": get_weights(layer),
                    "activation": "",
                }
            layers.append(layer_dict)

        model_dict["layers"] = layers
        return model_dict


    def save_model(model, filename):
        model_dict = save_model_json(model)
        with open(filename, "w") as outfile:
            json.dump(model_dict, outfile, cls=NumpyArrayEncoder, indent=4)


    save_model(model.model, f"./models/{model_name}.json")

    

# %%
