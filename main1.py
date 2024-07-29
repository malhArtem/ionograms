import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal

# df = pd.read_csv('data/1o.txt', sep=" ", header=None)
# df.columns = ["x", "y"]

with open('data/1o.txt') as f:
    data = f.readlines()
    coords = list(map(str.split, data))
x, y = zip(*coords)
x = list(map(int, x))
y = list(map(int, y))

coords = list(zip(x, y))

# window = signal.windows.hann(51)
window = np.hanning(len(x)) * np.hanning(len(x))[:, np.newaxis]
response = np.abs(np.fft.fftn(coords*window))
# response = 20 * np.log10(np.maximum(response, 1e-10))
plt.plot(response)
plt.show()
