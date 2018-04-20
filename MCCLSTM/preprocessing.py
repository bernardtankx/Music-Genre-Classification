import os

import librosa
import numpy as np
import scipy.signal

label_encoder = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7,
                 'reggae': 8, 'rock': 9}
n_chunks = 16
n_mels = 40
n_samples = 80
sr = 44100
hop_length = 1024
au_duration = 30
spec_len = int(np.rint(sr * au_duration / hop_length))


def chunk(input, chunk_size=80):
    """
    Convert input data in chunks of chunk_size. Excess are discarded.

    :param input: 2D array, data to be chunked.
    :param chunk_size: int, size of each chunk. Default is 80.
    :return: 3D array, number of chunks x length of rows x chunk_size
    """
    output = []
    for i in range(0, input.shape[1] // chunk_size * chunk_size, chunk_size):
        output.append(input[:, i:i + chunk_size])
    return np.array(output)


def normalize(state, data, mean_path='./models/mean.npy', std_path='./models/std.npy'):
    """
    Perform dynamic range compression and standardization of the data.
    If state is train, compute the mean and standard deviation of the data for standardization.
    If state is test, retrieve the train's mean and standard deviation for standardization.

    :param state: string, 'train' or 'test'.
    :param data: 3D array, data to be normalized.
    :param mean_path: path to mean file. Default is './models/mean.npy'.
    :param std_path: path to standard deviation file. Default is './models/std.npy'.
    :return: 3D array, normalized data.
    """
    data = np.log10(10000 * data + 1)
    if 'train' in state:
        mean = np.mean(data)
        std = np.std(data)
        np.save(mean_path, mean)
        np.save(std_path, std)
    else:
        mean = np.load(mean_path)
        std = np.load(std_path)
    data = (data - mean) / std
    return data


def spectrogram(file, sr=44100, win_length=2048, hop_length=1024, window=scipy.signal.blackmanharris(2048), n_mels=40):
    """
    Compute the mel-spectrogram.

    :param file: string, path to the file.
    :param sr: int, sampling rate. Default is 44100.
    :param win_length: int, window length. Default is 2048.
    :param hop_length: int, hop length. Default is 1024.
    :param window: scipy.signal, window used for the signal. Default is scipy.signal.blackmanharris(2048).
    :param n_mels: int, number of mels. Default is 40.
    :return: 2D array, mel-spectrogram
    """
    y, sr = librosa.core.load(file, sr=sr)

    length = sr * au_duration

    if len(y) > length:
        y = y[:length]
    elif len(y) < length:
        zeros = np.zeros(length - len(y))
        y = np.append(y, zeros)

    spec = np.abs(librosa.stft(y=y, hop_length=hop_length, window=window, win_length=win_length)) ** 2
    melspec = librosa.feature.melspectrogram(sr=sr, S=spec, hop_length=hop_length, n_mels=n_mels)
    return melspec


def extract(path, save_X, save_y):
    """
    Extract and store the chunked melspectrogram of the audio files contained in the path. Only files that end with
    '.au' are processed.

    :param path: string, path containing the audio files.
    :param save_X: string, path to save the training or test data.
    :param save_y: string, path to save the labels.
    """
    spec = []
    labels = []
    for root, dirs, files in os.walk(path):
        for file in sorted(files):
            if file.endswith('.au'):
                filepath = root + '/' + file
                print(filepath)
                label = (file.split('.'))[0]

                s = spectrogram(filepath)

                spec.append(s.reshape(1, *s.shape))
                labels.append(label_encoder[label])

    spec = np.concatenate(spec)
    spec = normalize(path, spec)

    X = []
    for i in range(len(spec)):
        c = chunk(spec[i, :, :])
        X.append(c.reshape(1, *c.shape, 1))

    X = np.concatenate(X)
    y = np.asarray(labels)

    np.save(save_X, X)
    np.save(save_y, y)


if __name__ == '__main__':
    extract(path='../train', save_X='./models/train_X.npy', save_y='./models/train_y.npy')
    extract(path='../test', save_X='./models/test_X.npy', save_y='./models/test_y.npy')
