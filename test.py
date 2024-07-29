import numpy as np
from scipy.signal import convolve
from scipy.signal.windows import hann

import filters_
from main import plot

# # Создание случайного сигнала
# signal = np.random.randn(1000)
# print(signal)
# # Создание окна Ханна
# window = hann(50)
#
# # Применение фильтра Ханна к сигналу с помощью окна Ханна
# filtered_signal = convolve(signal, window, mode='same') / sum(window)
# print(filtered_signal)
# # Визуализация исходного и отфильтрованного сигналов
import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(signal, label='Исходный сигнал')
# plt.plot(filtered_signal, label='Отфильтрованный сигнал')
# plt.legend()
# plt.show()

fig, sp = plt.subplots(1, 1)
# plot("data/1o.txt", color='blue', sp=sp)
plot("data/3o.txt", sp=sp, separ=True)

plt.show()