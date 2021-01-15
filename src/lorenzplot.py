import numpy as np
import scipy.signal as signal
import pandas as pd
from wfdb import processing
import matplotlib.pyplot as plt
import sys
import os


def butter_bandpass(lowcut, highcut, fs, order=4):
    # to gain paramaters of butterworth bandpass filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def apply_filter_to_signal(sig, lowcut, highcut, fs, order=4):
    # apply butterworth bandpass filter to the signal
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    filtered = signal.filtfilt(b, a, sig)
    return filtered


def plot_lorenz(rri):
    y = np.roll(rri, -1)[:-1]
    x = rri[:-1]
    plt.scatter(y, x)


def main():
    args = sys.argv

    if not os.path.exists(args[1]):
        exit()

    ecg_raw_df = pd.read_table(args[1], header=None, skiprows=13)[0]
    ecg_raw_df.head()

    Fs = 1000  # [Hz]
    # High cht freq: 30 [Hz]

    ecg_signal = ecg_raw_df.values
    ecg_signal = apply_filter_to_signal(ecg_signal, 0.3, 60, fs=Fs)
    ecg_signal = signal.resample(ecg_signal, (len(ecg_signal) // Fs) * 128)
    Fs = 128

    ecg_signal = ecg_signal[10*Fs:]

    conf = processing.XQRS.Conf(ref_period=0.4)
    xqrs = processing.XQRS(sig=ecg_signal, fs=Fs, conf=conf)
    xqrs.detect(learn=False)
    peaks = xqrs.qrs_inds / Fs
    RRI = np.diff(peaks)

    filename = os.path.basename(args[1]).split('.', 1)[0]
    csv_filepath = os.path.join('./data/csv', filename + '.csv')
    jpg_filepath = os.path.join('./data/lorenz', filename + '.jpg')

    np.savetxt(csv_filepath, RRI, delimiter=',', fmt='%f')

    plot_lorenz(RRI)

    plt.savefig(jpg_filepath, dpi=300)


if __name__ == '__main__':
    main()
