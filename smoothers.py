import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from statsmodels.nonparametric.kernel_regression import KernelReg
import statsmodels.api as sm

import filters_


def smooth_KernelReg(df: pd.DataFrame, tp='min'):
    points = []
    x = df['x'].unique()
    x.sort()
    for point in x:
        if tp == 'min':
            min_y = min(df.loc[df['x'] == point]['y'])
            points.append((point, min_y))
        elif tp == 'max':
            max_y = max(df.loc[df['x'] == point]['y'])
            points.append((point, max_y))
        elif tp == 'avr':
            avr_y = sum(df.loc[df['x'] == point]['y']) / len(df.loc[df['x'] == point]['y'])
            points.append((point, avr_y))


    new_df = pd.DataFrame(points, columns=['x', 'y'])
    kr = KernelReg(new_df['y'], new_df['x'], 'c')
    y_pred, y_std = kr.fit(new_df['x'])
    new_df['y'] = y_pred

    return new_df


def smooth_savgol(df: pd.DataFrame, window_length=35, polyorder=3, tp='min'):
    points = []
    x = df['x'].unique()
    x.sort()
    for point in x:
        if tp == 'min':
            min_y = min(df.loc[df['x'] == point]['y'])
            points.append((point, min_y))
        elif tp == 'max':
            max_y = max(df.loc[df['x'] == point]['y'])
            points.append((point, max_y))
        elif tp == 'avr':
            avr_y = sum(df.loc[df['x'] == point]['y']) / len(df.loc[df['x'] == point]['y'])
            points.append((point, avr_y))

    new_df = pd.DataFrame(points, columns=['x', 'y'])
    new_df['y'] = savgol_filter(new_df['y'], window_length, polyorder)

    return new_df


def smooth_lowess(df: pd.DataFrame, frac=0.1, tp='min'):
    if tp == 'all':
        new_df = df.copy()
        new_df = pd.DataFrame(sm.nonparametric.lowess(new_df['y'], new_df['x'], frac=frac, return_sorted=True, it=0), columns=['x', 'y'])

    else:
        points = []
        x = df['x'].unique()
        x.sort()
        for point in x:
            if tp == 'min':
                min_y = min(df.loc[df['x'] == point]['y'])
                points.append((point, min_y))
            elif tp == 'max':
                max_y = max(df.loc[df['x'] == point]['y'])
                points.append((point, max_y))
            elif tp == 'avr':
                avr_y = sum(df.loc[df['x'] == point]['y']) / len(df.loc[df['x'] == point]['y'])
                points.append((point, avr_y))

        new_df = pd.DataFrame(points, columns=['x', 'y'])
        new_df['y'] = sm.nonparametric.lowess(new_df['y'], new_df['x'], frac=frac, return_sorted=False, it=0)

    return new_df


def smooth_uniform_filer1d(df: pd.DataFrame, size=100, tp='min'):
    points = []
    x = df['x'].unique()
    x.sort()
    for point in x:
        if tp == 'min':
            min_y = min(df.loc[df['x'] == point]['y'])
            points.append((point, min_y))
        elif tp == 'max':
            max_y = max(df.loc[df['x'] == point]['y'])
            points.append((point, max_y))
        elif tp == 'avr':
            avr_y = sum(df.loc[df['x'] == point]['y']) / len(df.loc[df['x'] == point]['y'])
            points.append((point, avr_y))

    new_df = pd.DataFrame(points, columns=['x', 'y'])
    new_df['y'] = uniform_filter1d(new_df['y'], size=size, mode="wrap")

    return new_df


def smooth_moving_average(df: pd.DataFrame, n=11, tp='min'):
    points = []
    x = df['x'].unique()
    x.sort()
    for point in x:
        if tp == 'min':
            min_y = min(df.loc[df['x'] == point]['y'])
            points.append((point, min_y))
        elif tp == 'max':
            max_y = max(df.loc[df['x'] == point]['y'])
            points.append((point, max_y))
        elif tp == 'avr':
            avr_y = sum(df.loc[df['x'] == point]['y']) / len(df.loc[df['x'] == point]['y'])
            points.append((point, avr_y))

    new_df = pd.DataFrame(points, columns=['x', 'y'])
    # new_df['y'] = pd.concat([new_df['y'][:n-1], new_df['y'].rolling(window=n).mean().iloc[n - 1: -1], new_df['y'][-1:]])
    print(new_df)
    print(new_df['y'].rolling(window=n).mean()[n-1:].values)
    new_df = pd.DataFrame({'x': x,
                           'y': np.concatenate((new_df['y'].rolling(window=n).mean()[n//2:].values, new_df['y'].iloc[-(n//2):].values))})

    return new_df


def smooth_interval(df: pd.DataFrame, size=100, tp='min'):
    points = []
    x = df['x'].unique()
    x.sort()
    for point in x:
        if tp == 'min':
            min_y = min(df.loc[df['x'] == point]['y'])
            points.append((point, min_y))
        elif tp == 'max':
            max_y = max(df.loc[df['x'] == point]['y'])
            points.append((point, max_y))
        elif tp == 'avr':
            avr_y = sum(df.loc[df['x'] == point]['y']) / len(df.loc[df['x'] == point]['y'])
            points.append((point, avr_y))

    # new_points = []
    for i, point in enumerate(points[1:-1], start=1):
        interval = [points[i-1][1], points[i+1][1]] if points[i-1][1] < points[i+1][1] else [points[i+1][1], points[i-1][1]]

        if interval[0] <= point[1] <= interval[1]:
            # points.append(point)
            pass
        elif point[1] < interval[0]:
            points[i] = (point[0], interval[0])
        else:
            points[i] = (point[0], interval[1])
    # print(points)
    new_df = pd.DataFrame(points, columns=['x', 'y'])
    return new_df


def gistogram(df: pd.DataFrame, size=10, tp='min'):
    x = df['x'].unique()
    x.sort()
    new_dfs = []
    for i in range(len(x)//size + 1):
        target_x = x[i*size: (i+1)*size]
        y = df.loc[df['x'].isin(target_x)]['y'].sum()/len(df.loc[df['x'].isin(target_x)]['y'])
        print(target_x)
        print([y,]*len(target_x))
        new_dfs.append(pd.DataFrame({'x': target_x, 'y': [y,]*len(target_x)}))
    return pd.concat(new_dfs)


def derivative(df: pd.DataFrame):
    new_df = df.copy()
    diff = []
    for i in range(len(df['x'])-1):
        diff.append((df.iloc[i+1, -1] - df.iloc[i, -1])/(df.loc[i+1]['x']-df.loc[i]['x']))
    column_name = df.columns[-1] + "'"
    diff.append(0)
    new_df[column_name] = diff

    return new_df



def simplification(df: pd.DataFrame, n=10):
    new_df = df[1::n]
    print(new_df)
    new_df = new_df.reset_index(drop=True)
    return new_df



def percentile_smother(df: pd.DataFrame, percent=7, n=3):
    new_df = df.copy()
    xs = df['x'].unique()
    xs.sort()
    drop_ind = []
    for i in range(len(xs)//n):
        target_x = xs[i*n: (i+1)*n]
        target_y = new_df.loc[new_df['x'].isin(target_x)]['y']
        min_y = np.percentile(target_y, percent)
        drop_ind.extend(new_df.loc[(df['x'].isin(target_x)) & (new_df['y'] < min_y)].index)
    new_df = new_df.drop(drop_ind).reset_index(drop=True)
    return new_df


def median_filter_df(df: pd.DataFrame, size=3, tp="min"):
    points = []
    x = df['x'].unique()
    x.sort()
    for point in x:
        if tp == 'min':
            min_y = min(df.loc[df['x'] == point]['y'])
            points.append((point, min_y))
        elif tp == 'max':
            max_y = max(df.loc[df['x'] == point]['y'])
            points.append((point, max_y))
        elif tp == 'avr':
            avr_y = sum(df.loc[df['x'] == point]['y']) / len(df.loc[df['x'] == point]['y'])
            points.append((point, avr_y))

    points = median_filter(points, size=size)
    new_df = pd.DataFrame(points, columns=['x', 'y'])
    return new_df


def median_filter(points: list, size=3):
    new_points = []
    for i, v in enumerate(points[size//2:-(size//2)], start=size//2):
        x, y = zip(*points[i-size//2:i+size//2+1])
        y = list(y)
        y.sort()
        new_y = y[size//2+1]
        new_points.append((points[i][0], new_y))
    return new_points

