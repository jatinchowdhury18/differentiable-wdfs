# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from pathlib import Path

def getSampleRate(df):
    sample_rate_string = df.loc[4]
    sample_rate_string = str(sample_rate_string).split("#Sample rate:")
    Fs = sample_rate_string[1].split("Hz")[0]
    Fs = float(Fs)
    return Fs

def getDatasetSize(df):
    size = df.loc[5]
    size = size[0].split("#Samples: ")[1]
    size = float(size)
    return size

def createDataset(path, plot = False):
    info_df = pd.read_csv(path, skiprows= lambda x: x >8,header=None)
    Fs = getSampleRate(info_df)
    num_samples = getDatasetSize(info_df)

    output_df = pd.read_csv(path, header=9)
    output = output_df.to_numpy()

    time_remove_pre = 2.5 #seconds
    samp_trp = math.floor(time_remove_pre * Fs)
    
    dur_of_data = 14.3 #seconds
    samp_data_end = math.ceil((time_remove_pre + dur_of_data) * Fs)

    if plot:
        #plot cut lines
        plt.figure()
        plt.plot(output[:,0])
        plt.plot(output[:,1])
        plt.axvline(x=samp_trp, color="r")
        plt.axvline(x=samp_data_end, color="r")
        plt.title("Before, red lines are cut lines")
    
    output = output[samp_trp:samp_data_end,:]
    num_samples = len(output)

    if plot:
        plt.figure()
        plt.plot(output[:,0])
        plt.plot(output[:,1])
        plt.title("After")

    output_dict = {
        "dataset": output,
        "FS": Fs,
        "num_samples": num_samples
    }
    
    return output_dict
    

def get_data_path_for_diode(diode, BASE_DIR):
    path = Path(f'{BASE_DIR}/diode_dataset')

    if '1N4148' in diode.name:
        path = Path.joinpath(path, '1N4148')
    elif 'OA1154' in diode.name:
        path = Path.joinpath(path, 'OA1154')
    else:
        assert False, "No data available for this diode!"

    sub_folder = f'{diode.N_up}up{diode.N_down}down'
    return Path.joinpath(path, sub_folder)


def load_diode_data(diode, BASE_DIR, start_offset=0, csv_samples=-1, plot=False):
    data_path = get_data_path_for_diode(diode, BASE_DIR)

    total_num_samples = 0
    FS = 0
    all_data = pd.DataFrame()
    for csv_path in data_path.iterdir():

        R_val = float(csv_path.parts[-1].partition('k')[0])
        
        if R_val > 73:
            print(R_val)
            raw_data = createDataset(csv_path, plot=plot)
            FS = raw_data["FS"]

            N = raw_data["num_samples"] if csv_samples < 0 else csv_samples
            total_num_samples += N

            raw_data = raw_data["dataset"]
            
            x = raw_data[start_offset : start_offset + N, 0].astype(np.float32)
            R_data = np.ones_like(x) * (R_val * 1000.0)
            y_ref = raw_data[start_offset : start_offset + N, 1].astype(np.float32)

            csv_data = np.array([x, R_data, y_ref])
            csv_data_df = pd.DataFrame(data=csv_data)
            all_data = pd.concat([all_data, csv_data_df], axis=1)

        if R_val < 30:
            print(R_val)
            raw_data = createDataset(csv_path, plot=plot)
            FS = raw_data["FS"]

            N = raw_data["num_samples"] if csv_samples < 0 else csv_samples
            total_num_samples += N

            raw_data = raw_data["dataset"]
            
            x = raw_data[start_offset : start_offset + N, 0].astype(np.float32)
            R_data = np.ones_like(x) * (R_val * 1000.0)
            y_ref = raw_data[start_offset : start_offset + N, 1].astype(np.float32)

            csv_data = np.array([x, R_data, y_ref])
            csv_data_df = pd.DataFrame(data=csv_data)
            all_data = pd.concat([all_data, csv_data_df], axis=1)

    all_data_np = all_data.to_numpy()
    x = all_data_np[0]
    R_data = all_data_np[1]
    y_ref = all_data_np[2]

    return total_num_samples, FS, x, R_data, y_ref
# %%
# path = "/Users/chris/Desktop/git/differentiable-wdfs/diode_dataset/1N4148/1up1down/10.0k_4.7nF.csv"

# raw_data = createDataset(path)
# print(raw_data['dataset'])
# print(raw_data)


# %%
