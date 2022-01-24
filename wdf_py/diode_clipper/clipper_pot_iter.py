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
    "learning_rate": [1e-4, 1e-3, 1e-2],
    "training_number": [1, 2, 3],
    "epochs": [1001, 501, 501],
}

training_dict = {
    "n_layers": [2, 2, 2, 4, 4],
    "layer_size": [4, 8, 16, 4, 8],
}

for train_num in range(3):
    for num in range(5):
        if train_num < 1:
            if num < 4:
                continue

        n_layers = training_dict["n_layers"][num]
        layer_size = training_dict["layer_size"][num]
        print([n_layers, layer_size])
        diode = diode_1n4148_1u1d
        training_number = learning_rates_dict["training_number"][train_num]

        pretrained_model = f"{diode.name}_{n_layers}x{layer_size}_pretrained"
        model_name = f"{diode.name}_{n_layers}x{layer_size}_training_{training_number}"
        plots_dir = Path(f"./plots/{model_name}")

        assert (
            not plots_dir.exists()
        ), "Plots for this training run have already been created!"
        plots_dir.mkdir()

        C_val = 4.7e-9
        N, FS, x, R_data, y_ref = load_diode_data(diode, BASE_DIR)

        print(x.shape)
        print(y_ref.shape)

        batch_size = 2048
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

        plot_batch = 417
        # plt.plot(np.log(data_in_batched[plot_batch, :, 1]) - 10)
        plt.figure()
        plt.plot(data_in_batched[plot_batch, :, 0])
        plt.plot(data_target_batched[plot_batch, :, 0])
        plt.show()

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

                self.C.reset()

                for i in range(sequence_length):
                    self.Vs.set_voltage(input[:, i, 0:1])

                    self.Vs.set_resistance(input[:, i, 1:2])
                    self.P1.calc_impedance()

                    model_in = tf.concat(
                        (self.P1.reflected(), tf.math.log(self.P1.R)), axis=1
                    )
                    self.model.incident(tf.transpose(model_in, perm=[0, 2, 1]))
                    self.P1.incident(self.model.reflected())

                    output = wdf.voltage(self.C)
                    output_sequence = output_sequence.write(i, output)

                output_sequence = output_sequence.stack()
                return output_sequence

        with open(
            f"./models/pretrained/{pretrained_model}_model.json", "r"
        ) as read_file:
            model_json = json.load(read_file)
        # print(model_json)
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
            return tf.math.abs(target_min - pred_min) + tf.math.abs(
                target_max - pred_max
            )

        mse_loss = tf.keras.losses.MeanSquaredError()
        loss_func = lambda target, pred: mse_loss(target, pred) + esr_loss(target, pred)

        # optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-9)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=float(learning_rates_dict["learning_rate"][train_num])
        )

        def plot_target_pred(target, predicted, epoch):
            plt.figure()
            plt.plot(target[:batch_size], label="Target")
            plt.plot(predicted[:batch_size], "--", label="Predicted")
            plt.xlabel("Time [samples]")
            plt.ylabel("Voltage")
            plt.title(
                f"Diode Clipper ({diode.name}, {n_layers}x{layer_size}), Epoch {epoch}"
            )
            plt.legend(loc="lower left")

            plt.savefig(f"./{plots_dir}/epoch_{epoch}.png")
            plt.close()

        skip_samples = 50  # skip the first few samples to let state build up
        history = {"loss": [], "mse": [], "esr": []}

        print(data_in.shape)
        print(data_in_batched.shape)
        print(data_target.shape)
        print(data_target_batched.shape)

        for epoch in tqdm(range(learning_rates_dict["epochs"][train_num])):
            with tf.GradientTape() as tape:
                outs = tf.transpose(
                    model.forward(data_in_batched)[..., 0], perm=[1, 0, 2]
                )
                loss = loss_func(
                    outs[:, skip_samples:, :], data_target_batched[:, skip_samples:, :]
                )

            history["loss"].append(loss)
            history["mse"].append(
                mse_loss(
                    outs[:, skip_samples:, :], data_target_batched[:, skip_samples:, :]
                )
            )
            history["esr"].append(
                esr_loss(
                    outs[:, skip_samples:, :], data_target_batched[:, skip_samples:, :]
                )
            )

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if epoch % 50 == 0:
                print(f"\nCheckpoint (Epoch = {epoch}):")
                print(f"    Loss: {loss}")
                target = data_target_batched[plot_batch, skip_samples:, 0]
                pred = outs[plot_batch, skip_samples:, 0]
                plot_target_pred(target, pred, epoch)

        print(f"\nFinal Results:")
        print(f"    Loss: {loss}")
        with open(f"./histories/{model_name}_history.pkl", "wb") as f:
            pickle.dump(history, f)

        outs = tf.transpose(model.forward(data_in_batched)[..., 0], perm=[1, 0, 2])

        target = data_target_batched[plot_batch, skip_samples:, 0]
        pred = outs[plot_batch, skip_samples:, 0]
        plot_target_pred(target, pred, "final")
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
