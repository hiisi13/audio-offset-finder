import argparse

import librosa
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def find_offset(within_file, find_file, window):
    y_within, sr_within = librosa.load(within_file, sr=None)
    y_find, _ = librosa.load(find_file, sr=sr_within)

    c = signal.correlate(y_within, y_find[:sr_within*window], mode='valid', method='fft')
    peak = np.argmax(c)
    offset = round(peak / sr_within, 2)

    fig, ax = plt.subplots()
    ax.plot(c)
    fig.savefig("cross-correlation.png")

    return offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--find-offset-of', metavar='audio file', type=str, help='Find the offset of file')
    parser.add_argument('--within', metavar='audio file', type=str, help='Within file')
    parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of a target audio')
    args = parser.parse_args()
    offset = find_offset(args.within, args.find_offset_of, args.window)
    print(f"Offset: {offset}s" )


if __name__ == '__main__':
    main()