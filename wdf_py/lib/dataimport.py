# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

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



def createDataset(path):
    info_df = pd.read_csv(path, skiprows= lambda x: x >8,header=None)
    Fs = getSampleRate(info_df)
    num_samples = getDatasetSize(info_df)

    output_df = pd.read_csv(path, header=9)
    output = output_df.to_numpy()

    time_remove_pre = 2.5 #seconds
    samp_trp = math.floor(time_remove_pre * Fs)
    
    dur_of_data = 12.3 #seconds
    samp_data_end = math.ceil((time_remove_pre + dur_of_data) * Fs)

    #plot cut lines
    plt.plot(output[:,0])
    plt.plot(output[:,1])
    plt.axvline(x=samp_trp, color="r")
    plt.axvline(x=samp_data_end, color="r")
    plt.title("Before, red lines are cut lines")
    
    plt.figure()
    output = output[samp_trp:samp_data_end,:]
    num_samples = len(output)
    plt.plot(output[:,0])
    plt.plot(output[:,1])
    plt.title("After")

    output_dict = {
        "dataset": output,
        "FS": Fs,
        "num_samples": num_samples
    }
    


    return output_dict
    
# %%
# path = "/Users/chris/Desktop/git/differentiable-wdfs/diode_dataset/1N4148/1up1down/10.0k_4.7nF.csv"

# raw_data = createDataset(path)
# print(raw_data['dataset'])
# print(raw_data)


# %%
