from typing import Union

import matplotlib.axes
import numpy as np
import pandas as pd
from matplotlib import pyplot as pt

from filters_ import plot_df, MedianFilter2d
from separators import FSeparator, EFSeparator
from smoothers import simplification, median_filter_df


def plot_borders(x, y, z, color="black"):
    min_x = 0
    max_x = len(x)
    min_y = 0
    max_y = len(y)

    for index, value in enumerate(z):
        if value > 3.5e-10:
            min_x = index
            min_y = index
            break

    for index, value in enumerate(z[::-1]):
        if value > 3.5e-10:
            max_x = -index
            max_y = -index
            break

    pt.axvline(x=x[min_x], color=color,
               ymin=y[min_x] / 1000_000 - 0.1,
               ymax=y[min_x] / 1000_000 + 0.1
               )
    pt.axvline(x=x[max_x], color=color,
               ymin=y[max_x] / 1000_000 - 0.1,
               ymax=y[max_x] / 1000_000 + 0.1
               )

    pt.axhline(y=y[min_y], color=color,
               xmin=x[min_y] / 15000 - 0.06,
               xmax=x[min_y] / 15000 + 0.06
               )
    pt.axhline(y=y[max_y], color=color,
               xmin=x[max_y] / 15000 - 0.06,
               xmax=x[max_y] / 15000 + 0.06
               )


def find_hpF2(layer_F2: pd.DataFrame):
    FoF2 = max(layer_F2['x'])
    fp = FoF2 * 0.834
    m, M = layer_F2.loc[layer_F2['x'] <= fp], layer_F2.loc[layer_F2['x'] > fp]
    nearst = m['x'].iloc[-1] if abs(m['x'].iloc[-1] - fp) <= abs(M['x'].iloc[0] - fp) else M['x'].iloc[0]
    return layer_F2.loc[layer_F2['x'] == nearst]['y'].min()


def plot(file_path: Union[tuple, str], color='red', cmap='hot', filtering=None, sp: matplotlib.axes.Axes = pt,
         separ=None, type_='o', post_filter=None):
    sp.grid(visible=True)
    coords = []
    if isinstance(file_path, str):
        with open(file_path) as f:
            data = f.readlines()
            coords = list(map(str.split, data))
    else:
        for path in file_path:
            with open(path) as f:
                data = f.readlines()
                coords.extend(list(map(str.split, data)))

    coords.sort()

    new_coords = []
    for coord in coords:
        if len(coord) > 2 and int(coord[2]) >= 17:
            new_coords.append(coord)
    if len(coords[0]) > 2:
        coords = new_coords
        x, y, a = zip(*coords)

    else:
        x, y = zip(*coords)

    x = list(map(int, x))
    y = list(map(int, y))

    df = pd.DataFrame({
        'x': x,
        'y': y,
    })

    if filtering is not None:
        for filter_ in filtering:
            df = filter_(df)

    if separ is not None:
        F_layer, E_layer, Es_layer, sublayer = separ(df)
        print(F_layer)
        if post_filter is not None:
            for filter_ in post_filter:
                F_layer = filter_(F_layer)
                if E_layer is not None:
                    E_layer = filter_(E_layer)
                if Es_layer is not None:
                    Es_layer = filter_(Es_layer)

        plot_df(F_layer, color=color, sp=sp)

        if E_layer is not None:
            plot_df(E_layer, color="green", sp=sp)

        if Es_layer is not None:
            plot_df(Es_layer, color="blue", sp=sp)
        print(sublayer)
        plot_df(sublayer, color="gray", sp=sp)

        return F_layer, E_layer, Es_layer, sublayer

    else:
        sp.scatter(df['x'], df['y'], s=1, c=color)

    sp.set(xlabel="Частота",
           ylabel="Высота",
           xlim=(0, 13000),
           ylim=(0, 900000),
           xticks=np.arange(0, 13000, 1000),
           yticks=np.arange(0, 900_000, 50_000))


def ionogram_processing(o_path, x_path, pre_process=None, sp: matplotlib.axes.Axes = pt, separ=None, post_process=None):
    F_layer, E_layer, Es_layer, sublayer = plot(o_path, sp=sp, separ=separ, filtering=pre_process, type_='o',
                                                post_filter=post_process)
    X_layer = \
    plot(x_path, sp=sp, separ=separ, filtering=pre_process, type_='x', color='blue', post_filter=post_process)[0]
    param = {}

    separator = FSeparator(smoothers=(median_filter_df,), simplificator=simplification)
    print(F_layer)
    F1_layer, F2_layer = separator(F_layer)
    print(F1_layer)
    print(F2_layer)
    F2_layer: pd.DataFrame
    sp.axvline(x=F2_layer.iloc[0]['x'], color="black")

    param['fmin'] = min(E_layer['x']) if E_layer is not None else min(F_layer['x'])
    param['foE'] = max(E_layer['x']) if E_layer is not None else None
    param['foF1'] = max(F1_layer['x']) if F1_layer is not None else None
    param['foF2'] = max(F_layer['x'])
    param['FbEs'] = min(F_layer['x'])
    param['h`E'] = min(E_layer['y']) if E_layer is not None else None
    param['h`Es'] = min(Es_layer['y']) if Es_layer is not None else None
    param['h`F'] = min(F_layer['y'])
    param['h`F2'] = min(F2_layer['y'])
    param['fxF2'] = max(X_layer['x'])

    param['hpF2'] = find_hpF2(F2_layer)

    for k, v in param.items():
        if v is not None:
            if k[0] == 'f' or k[0] == 'F':
                print(k, ': ', v, 'МГц')
            else:
                print(k, ': ', v, 'м')


"""Найти частоты и высоты разных слоев (где начинаются)"""

if __name__ == '__main__':
    f, plts = pt.subplots(1, 1)
    ionogram_processing("data/4o.txt", "data/4x.txt", sp=plts, separ=EFSeparator(), pre_process=(MedianFilter2d(size=3),))

    pt.show()
