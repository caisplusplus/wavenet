import numpy as np
import glob
from scipy.io import wavfile


def normalize(data):
    # Preprocessing step to make the audio easier for our network
    # to learn
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def make_batch(path, time_steps):
    data = wavfile.read(path)[1]

    if len(data) < time_steps:
        return None

    # Cut the input to fit our set length
    # This allows us to have many samples that are the same length
    data = data[:time_steps + 1]

    data_ = normalize(data)
    # data_f = np.sign(data_) * (np.log(1 + 255*np.abs(data_)) / np.log(1 + 255))

    # Each audio samples is a byte (aka 256 unique values)
    bins = np.linspace(-1, 1, 256)
    # Quantize inputs.
    inputs = np.digitize(data_[0:-1], bins, right=False) - 1
    inputs = bins[inputs][None, :, None]

    # Encode targets as ints.
    # (notice that we are not using the first element of the input data)
    targets = (np.digitize(data_[1::], bins, right=False) - 1)[None, :]
    return inputs, targets

def make_batch_single(path, time_steps):
    inputs, targets = make_batch(path, time_steps)
    return [inputs], [targets]

def make_batch_all(path, take_count, time_steps):
    # Just read in all wav files by recursively searching directory.
    X, Y = [], []
    for filename in list(glob.iglob(path + '/*/*.wav'))[:take_count]:
        inputs, targets = make_batch(filename, time_steps)
        X.append(inputs)
        Y.append(targets)

    return X, Y
