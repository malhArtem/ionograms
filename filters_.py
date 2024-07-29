from abc import abstractmethod

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import uniform_filter
from scipy.signal import windows, sepfir2d
from scipy.stats import gaussian_kde


class Filter:
    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class MedianFilter2d(Filter):
    def __init__(self, size=3):
        self.size = size

    def __call__(self, df: pd.DataFrame):
        # переводим координаты в изображение
        B = coords_to_image(df)

        # применяем медианный фильтр
        filtered = median_filter2d(B.to_numpy(), kernel_size=self.size)

        # переводим изображение обратно в координаты
        new_df = matrix_to_coords(filtered, B.columns, B.index)

        return new_df


class GaussianFilter(Filter):
    def __init__(self, power=3e-10):
        self.power = power

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        xy = np.vstack((new_df['x'].values, new_df['y'].values))
        z = gaussian_kde(xy)(xy)
        outlier_indices = np.where(z < self.power)[0]
        print(outlier_indices)
        no_outliers = new_df.drop(outlier_indices, axis='index')

        return no_outliers


class LowFilter(Filter):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        nyquist = 0.5 * 100
        low = 10 / nyquist
        b, a = signal.butter(1, low, btype='low')
        filtered = signal.lfilter(b, a, df)
        return pd.DataFrame(filtered, columns=['x', 'y'])


class HannFilter(Filter):
    def __init__(self, size=15, tp='all'):
        self.size = size
        self.tp = tp

    def __call__(self, df: pd.DataFrame):
        if self.tp == 'all':
            new_df = df.copy()

        else:
            points = []
            x = df['x'].unique()
            x.sort()
            for point in x:
                if self.tp == 'min':
                    min_y = min(df.loc[df['x'] == point]['y'])
                    points.append((point, min_y))
                elif self.tp == 'max':
                    max_y = max(df.loc[df['x'] == point]['y'])
                    points.append((point, max_y))
                elif self.tp == 'avr':
                    avr_y = sum(df.loc[df['x'] == point]['y']) / len(df.loc[df['x'] == point]['y'])
                    points.append((point, avr_y))

            new_df = pd.DataFrame(points, columns=['x', 'y'])

        hann_window = windows.hann(self.size)
        hann_window = hann_window / hann_window.sum()
        # Применение фильтра Хэнна
        filtered_data = np.convolve(hann_window, new_df['y'], mode='valid')
        sli = int((self.size - 1) / 2)
        new_df = pd.DataFrame({'x': new_df['x'].values[sli:-sli], 'y': filtered_data})

        return new_df


class HannFilter2d(Filter):
    def __init__(self, size=5):
        self.size = size

    def __call__(self, df: pd.DataFrame):
        hann_window = windows.hann(self.size)
        hann_window = hann_window / hann_window.sum()
        B = coords_to_image(df)
        # print(B[1106].to_string())
        # Применение фильтра Хэнна
        # filtered_data = (hann_window, df['y'], mode='valid')
        convolved = sepfir2d(B.to_numpy(), hann_window, hann_window)
        np.set_printoptions(threshold=10000)
        filtered_data = matrix_to_coords(convolved, B.columns, B.index, threshold=True)
        # sli = int((size-1)/2)
        # new_df = pd.DataFrame({'x': df['x'].values[sli:-sli], 'y': filtered_data})

        return filtered_data


class HammingFilter2d(Filter):
    def __init__(self, size=5):
        self.size = size

    def __call__(self, df: pd.DataFrame):
        hamming_window = windows.hamming(self.size)
        hamming_window = hamming_window / hamming_window.sum()
        B = coords_to_image(df)
        # print(B[1106].to_string())
        # Применение фильтра Хэнна
        # filtered_data = (hann_window, df['y'], mode='valid')
        convolved = sepfir2d(B.to_numpy(), hamming_window, hamming_window)
        filtered_data = matrix_to_coords(convolved, B.columns, B.index, threshold=True)
        # sli = int((size-1)/2)
        # new_df = pd.DataFrame({'x': df['x'].values[sli:-sli], 'y': filtered_data})

        return filtered_data


class KaiserFilter2d(Filter):
    def __init__(self, size=5, beta=0):
        self.size = size
        self.beta = beta

    def __call__(self, df: pd.DataFrame):
        kaiser_window = windows.kaiser(self.size, beta=self.beta)
        kaiser_window = kaiser_window / kaiser_window.sum()
        B = coords_to_image(df)
        # print(B[1106].to_string())
        # Применение фильтра Хэнна
        # filtered_data = (hann_window, df['y'], mode='valid')
        convolved = sepfir2d(B.to_numpy(), kaiser_window, kaiser_window)
        filtered_data = matrix_to_coords(convolved, B.columns, B.index, threshold=True)
        # sli = int((size-1)/2)
        # new_df = pd.DataFrame({'x': df['x'].values[sli:-sli], 'y': filtered_data})

        return filtered_data


# class ZScoreFilter(Filter):
#     def __init__(self, threshold_z=1.6):
#         self.threshold_z = threshold_z
#
#     def __call__(self, df: pd.DataFrame):
#         z = np.abs(stats.zscore(df))
#         # print(z)
#         outlier_indices = np.where(z > self.threshold_z)[0]
#         no_outliers = df.drop(outlier_indices)
#         return no_outliers


class MovingAvrFilter2d(Filter):
    def __init__(self, size=5, threshold=10 / 25):
        self.size = size
        self.threshold = threshold

    def __call__(self, df: pd.DataFrame):
        B = coords_to_image(df)
        filtered = uniform_filter(B.to_numpy(), size=self.size)
        new_df = matrix_to_coords(filtered, B.columns, B.index, threshold=self.threshold, old_coords=df)
        return new_df


# def hann_filter_2d(df: pd.DataFrame, size=5):
#     hann_window = windows.hann(size)
#     hann_window = hann_window / hann_window.sum()
#     B = coords_to_image(df)
#     # print(B[1106].to_string())
#     # Применение фильтра Хэнна
#     # filtered_data = (hann_window, df['y'], mode='valid')
#     convolved = sepfir2d(B.to_numpy(), hann_window, hann_window)
#     np.set_printoptions(threshold=10000)
#     print(convolved[convolved.nonzero()])
#     filtered_data = matrix_to_coords(convolved, B.columns, B.index)
#     # sli = int((size-1)/2)
#     # new_df = pd.DataFrame({'x': df['x'].values[sli:-sli], 'y': filtered_data})
#
#     return filtered_data


# def low_filter(df: pd.DataFrame):
#     nyquist = 0.5 * 100
#     low = 10 / nyquist
#     b, a = signal.butter(1, low, btype='low')
#     filtered = signal.lfilter(b, a, df)
#     return pd.DataFrame(filtered, columns=['x', 'y'])


# def zscore_filter(df: pd.DataFrame, threshold_z=1.6):
#     z = np.abs(stats.zscore(df))
#     # print(z)
#     outlier_indices = np.where(z > threshold_z)[0]
#     no_outliers = df.drop(outlier_indices)
#     return no_outliers


# def gaussian_filter(df: pd.DataFrame, power=3e-10):


# def hann_filter(df: pd.DataFrame, tp='all', size=15):
#     if tp == 'all':
#         new_df = df.copy()
#
#     else:
#         points = []
#         x = df['x'].unique()
#         x.sort()
#         for point in x:
#             if tp == 'min':
#                 min_y = min(df.loc[df['x'] == point]['y'])
#                 points.append((point, min_y))
#             elif tp == 'max':
#                 max_y = max(df.loc[df['x'] == point]['y'])
#                 points.append((point, max_y))
#             elif tp == 'avr':
#                 avr_y = sum(df.loc[df['x'] == point]['y']) / len(df.loc[df['x'] == point]['y'])
#                 points.append((point, avr_y))
#
#         new_df = pd.DataFrame(points, columns=['x', 'y'])
#
#
#     hann_window = windows.hann(size)
#     hann_window = hann_window / hann_window.sum()
#     # Применение фильтра Хэнна
#     filtered_data = np.convolve(hann_window, new_df['y'], mode='valid')
#     sli = int((size-1)/2)
#     new_df = pd.DataFrame({'x': new_df['x'].values[sli:-sli], 'y': filtered_data})
#
#     return new_df


def layer_max(layer):
    layer_x, layer_y = zip(*layer)
    return max(layer_y)


def plot_df(layer: pd.DataFrame, sp, c=None, color=None):
    if not layer.empty:
        sp.scatter(layer['x'], layer['y'], s=20, c=c, color=color)


def coords_to_image(df: pd.DataFrame):
    xx = df['x'].unique()
    yy = np.arange(df['y'].min(), df['y'].max(), 1500)
    matrix = pd.DataFrame(0, index=yy, columns=xx)
    for index, row in df.iterrows():
        matrix.loc[row['y'], row['x']] = 1
    return matrix


def image_to_coords(df: pd.DataFrame):
    new_df = pd.DataFrame(columns=['x', 'y'])
    for x in df.columns:
        for y in df.index:
            if df.loc[y, x] == 1:
                new_df = pd.concat([new_df, pd.DataFrame([x, y], columns=['x', 'y'])], ignore_index=True)

    return new_df


def matrix_to_coords(matrix: np.ndarray, x, y, threshold=None):
    mat = matrix[matrix.nonzero()]
    mat = mat[~np.isnan(mat)]
    med = np.median(mat)

    matrix[np.isnan(matrix)] = 0
    if threshold is True:
        matrix[matrix < med * 3 / 2] = 0

    elif threshold is not None:
        matrix[matrix < threshold] = 0
    yy, xx = matrix.nonzero()
    yy = np.array(y[yy])
    xx = np.array(x[xx])

    df1 = pd.DataFrame({"x": xx, "y": yy})

    return df1


def median_filter2d(image, kernel_size):
    """Двумерный медианный фильтр.

    Params:
      image: исходное изображение
      kernel_size: размер ядра

    Returns:
      изображение после применения медианного фильтра
    """

    # Добавим отступы к изображению для обработки краев
    image = np.pad(image, kernel_size // 2, mode='edge')

    # Инициализируем результирующее изображение
    filtered_image = np.zeros_like(image)

    # Проходим по каждому пикселю изображения
    for i in range(kernel_size // 2, image.shape[0] - kernel_size // 2):
        for j in range(kernel_size // 2, image.shape[1] - kernel_size // 2):
            # Получим окно ядра
            window = image[i - kernel_size // 2:i + kernel_size // 2 + 1,
                     j - kernel_size // 2:j + kernel_size // 2 + 1]

            # Вычислим медиану в окне ядра
            median = np.median(window)

            # Запишем медиану в результирующее изображение
            filtered_image[i][j] = median

    # Обрежем отступы
    return filtered_image[kernel_size // 2:-kernel_size // 2, kernel_size // 2:-kernel_size // 2]


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Двумерная свертка изображения с ядром.

    Args:
        image (ndarray): Исходное изображение.
        kernel (ndarray): Ядро свертки.

    Returns:
        ndarray: Результат свертки.
    """

    # Проверим размеры изображения и ядра
    if image.shape[0] < kernel.shape[0] or image.shape[1] < kernel.shape[1]:
        raise ValueError("Изображение должно быть больше ядра.")

    # Добавим нули к изображению для обработки краев
    image = np.pad(image, ((kernel.shape[0] // 2, kernel.shape[0] // 2), (kernel.shape[1] // 2, kernel.shape[1] // 2)),
                   'constant')

    # Инициализируем результирующее изображение
    result = np.zeros_like(image)

    # Выполним свертку
    for i in range(image.shape[0] - kernel.shape[0] + 1):
        for j in range(image.shape[1] - kernel.shape[1] + 1):
            result[i, j] = np.sum(image[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel)

    # Вернем результирующее изображение
    return result


#
# def mov_avr_2d(df: pd.DataFrame, size=5, threshold=10/25):
#     B = coords_to_image(df)
#     filtered = uniform_filter(B.to_numpy(), size=size)
#     new_df = matrix_to_coords(filtered, B.columns, B.index, threshold=threshold, old_coords=df)
#     return new_df


# def median_filter_2d(df: pd.DataFrame, size=3):
#     B = coords_to_image(df)
#     filtered = medfilt2d(B.to_numpy(), kernel_size=size)
#     new_df = matrix_to_coords(filtered, B.columns, B.index)
#     return new_df


def hann2d(M: int) -> np.ndarray:
    """
    Создает двумерное окно Ханна.

    Args:
      M: количество строк в окне.
      N: количество столбцов в окне.

    Returns:
      Двумерный массив, представляющий окно Ханна.
    """

    # Создать одномерное окно Ханна
    hann = 0.5 * (1 + np.cos(np.pi * np.arange(M) / (M - 1)))

    # Создать двумерное окно Ханна, перемножив строки и столбцы
    hann2d = np.outer(hann, hann)

    return hann2d
