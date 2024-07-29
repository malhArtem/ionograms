from abc import abstractmethod

import numpy as np
import pandas as pd

from smoothers import simplification, derivative, median_filter_df


class Separator:
    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> tuple[pd.DataFrame]:
        pass


class EFSeparator(Separator):
    """Разделяет ионограмму на слои F, E, Es(может не существовать) и шум"""


    def __call__(self, df: pd.DataFrame):
        """
        Разделяет ионограмму на слои F, E, Es(может не существовать) и шум
        :param df: датафрейм точек, который нужно разделить
        :return: слой F, слой E, слой Es(None), шум(то что не вошло в другие слои)
        """
        coords = zip(df['x'], df['y'])
        layers, sublayer = clusterisation(coords, eps=300)
        F_layer = []
        E_layers = []
        for layer_ in layers:
            layer = pd.DataFrame(layer_, columns=['x', 'y'])
            if min(layer['y']) > 280_000:
                sublayer.extend(layer_)
            elif max(layer['y']) < 210_000 and min(layer['y']) > 90_000:
                E_layers.append(layer)
            elif min(layer['y']) > 140_000:
                F_layer.append(layer)

            else:
                rem_layer, new_sublayer, new_layers = separate_EF(layer)
                F_layer.append(rem_layer)
                sublayer.extend(new_sublayer)

                for new_layer_ in new_layers:
                    new_layer = pd.DataFrame(new_layer_, columns=['x', 'y'])
                    if min(new_layer['y']) > 280_000:
                        sublayer.extend(new_layer_)
                    elif max(new_layer['y']) < 210_000 and min(new_layer['y']) > 90_000:
                        E_layers.append(new_layer)
                    elif min(layer['y']) > 140_000:
                        F_layer.append(new_layer)
                    else:
                        F_layer.append(new_layer)

        E_layer = None
        Es_layer = None

        if len(E_layers) == 2:
            if len(E_layers[0]) > len(E_layers[1]):
                E_layer, Es_layer = E_layers[0], E_layers[1]
            else:
                E_layer, Es_layer = E_layers[1], E_layers[0]

        elif len(E_layers) == 1:
            E_layer = E_layers[0]

        elif len(E_layers) != 0:
            E_layer = pd.concat(E_layers)

        # print(F_layer[0])
        # print(sublayer[-1])
        return pd.concat(F_layer), E_layer, Es_layer, pd.DataFrame(sublayer, columns=['x', 'y'])


class FSeparator(Separator):
    """
    Разделяет слой F на F1 и F2, путем поиска перегиба
    """

    def __init__(self, smoothers: tuple = None, simplificator=None):
        self.smoothers = smoothers
        self.simplificator = simplificator

    def __call__(self, df_: pd.DataFrame):
        print(df_)
        """
        :param df_: слой F
        :return: слои F1 и F2
        """
        df = df_.copy()
        for smoother in self.smoothers:
            df = smoother(df)

        target_df = self.simplificator(df)
        target_df = derivative(target_df)
        target_df = derivative(target_df)
        print(target_df)
        target_df = target_df[len(target_df) // 5:-len(target_df) // 5]

        target_df.reset_index(inplace=True, drop=True)

        x1, x2, x3, type_ = search_inflection(target_df)
        if x1 is None and self.simplificator is not None:
            target_df = derivative(df)
            target_df = derivative(target_df)
            target_df = target_df[len(target_df) // 5:-len(target_df) // 5]
            target_df.reset_index(inplace=True, drop=True)
            x1, x2, x3, type_ = search_inflection(target_df)
            if x1 is None:
                return None, df_


        sep = (x3 + x1) / 2
        if self.simplificator is None:
            sep = x2

        else:
            if type_ == "high":
                max_y = df[df["x"].between(x1, x3)].idxmax()['y']
                sep = df.iloc[max_y]["x"]

            elif type_ == "low":
                target_df = df[df["x"].between(x1, x3)]
                target_df.reset_index(inplace=True, drop=True)
                target_df = derivative(target_df)
                target_df = derivative(target_df)
                for i, v in enumerate(target_df["y'"]):
                    if (v >= 0 and target_df.iloc[i]["y''"] > 0) and i < len(target_df) - 2 and (
                            target_df.iloc[i + 1]["y''"] < 0):
                        sep = target_df.iloc[i]["x"]
                        break
            else:
                target_df = df[df["x"].between(x1, x3)]
                target_df.reset_index(inplace=True, drop=True)
                target_df = derivative(target_df)
                target_df = derivative(target_df)
                for i, v in enumerate(target_df["y'"]):
                    if v < 0.01:
                        sep = target_df.iloc[i]["x"]
                        break
        F1 = df.loc[df["x"] <= sep]
        F2 = df.loc[df["x"] > sep]
        return F1, F2


def search_inflection(target_df: pd.DataFrame):
    for i, v in enumerate(target_df["y'"]):
        if v >= 0 and i < len(target_df) - 2 and target_df.iloc[i + 1]["y'"] < 0:
            x1 = target_df.iloc[i]["x"]
            x2 = target_df.iloc[i + 1]["x"]
            x3 = target_df.iloc[i + 2]["x"]
            print("Сильный перегиб")
            return x1, x2, x3, "high"

    for i, v in enumerate(target_df["y'"]):
        if (v >= 0 and target_df.iloc[i]["y''"] > 0) and i < len(target_df) - 2 and (
                target_df.iloc[i + 1]["y''"] < 0):
            x1 = target_df.iloc[i]["x"]
            x2 = target_df.iloc[i + 1]["x"]
            x3 = target_df.iloc[i + 2]["x"]
            print("Слабый перегиб")
            return x1, x2, x3, "low"

    for i, v in enumerate(target_df["y'"]):
        if (v >= 0 and target_df.iloc[i]["y''"] < 0) and target_df.iloc[i]["y'"] < 0.01:
            x1 = target_df.iloc[i - 1]["x"]
            x2 = target_df.iloc[i]["x"]
            x3 = target_df.iloc[i + 1]["x"] if i != len(target_df) - 1 else target_df.iloc[i]["x"]
            print("без перегиба")
            return x1, x2, x3, "not"

    return None, None, None, None


def clusterisation(coords, ratio=70, eps=140):
    segments = []
    segment = []
    while coords:
        new_coords = []
        for i, point in enumerate(coords):

            if not segment:
                segment.append(point)
            else:
                if segment[-1][0] == point[0] and abs(segment[-1][1] - point[1]) < 1500 * 5:
                    segment.append(point)
                elif segment[-1][0] != point[0]:
                    segments.append(segment)
                    segment = []
                    segment.append(point)
                else:
                    new_coords.append(point)

        coords = new_coords

    segments.sort()
    layers = []
    sublayer = []

    while segments:
        layer = []
        new_segments = []
        for i, segment in enumerate(segments):
            if not layer:
                layer.extend(segment)
            else:
                dists = []

                for point1 in segment:
                    for point2 in reversed(layer):
                        if abs(point1[0] - point2[0]) / 1.5 < eps:
                            dist = np.sqrt(
                                ((point1[0] - point2[0])) ** 2 + ((point1[1] - point2[1]) / ratio) ** 2)
                            dists.append(dist)
                        else:
                            break

                if dists and min(dists) < eps:
                    layer.extend(segment)

                else:
                    new_segments.append(segment)

        if len(layer) < 100:
            sublayer.extend(layer)
        else:
            layers.append(layer)

        segments = new_segments

    return layers, sublayer




def separate_EF(layer: pd.DataFrame):
    sep_part = layer[layer['x'] < 4000]
    rem_layer = layer[layer['x'] >= 4000]
    sep_part = zip(sep_part['x'], sep_part['y'])
    layers, sublayer = clusterisation(sep_part, eps=60)
    return rem_layer, sublayer, layers

def separate_layers(coords):
    layer1 = []
    layer2 = []
    layer3 = []
    layer4 = []

    x_equals = {}

    for point in coords:
        x_equals.setdefault(point[0], []).append(point[1])

    for key, values in x_equals.items():
        cur_layer = layer1
        sum_ = 0
        # for i in range(len(values)-1):
        #     sum_ += values[i+1] - values[i]
        #
        #
        #
        # average = sum_ / (len(values)-1)
        # print(average)

        for i in range(1, len(values)):
            if values[i] > values[i-1] + 1500 * 60:
                if cur_layer is layer1:
                    cur_layer = layer2

                elif cur_layer is layer2:
                    cur_layer = layer3

                else:
                    cur_layer = layer4

            while cur_layer != [] and values[i] > layer_max(cur_layer) + 1500 * 30:
                if cur_layer is layer1:
                    cur_layer = layer2

                elif cur_layer is layer2:
                    cur_layer = layer3

                else:
                    cur_layer = layer4
                    break

            cur_layer.append([key, values[i]])

    layers, sublayer = separate_layers_second_pass_v2(layer1, layer2, layer3)
    return layers, sublayer

def separate_layers_v2(coords):
    layers, sublayer = clusterisation(coords)
    F_layer = []
    E_layers = []
    for layer in layers:
        layer = pd.DataFrame(layer, columns=['x', 'y'])
        if min(layer['y']) > 280_000:
            pass
        elif max(layer['y']) < 210_000 and min(layer['y']) > 90_000:
            E_layers.append(layer)
        elif min(layer['y']) > 140_000:
            F_layer.append(layer)

        else:
            rem_layer, new_sublayer, new_layers = separate_EF(layer)
            F_layer.append(rem_layer)
            sublayer.extend(new_sublayer)

            for new_layer in new_layers:
                new_layer = pd.DataFrame(new_layer, columns=['x', 'y'])
                if min(new_layer['y']) > 280_000:
                    pass
                elif max(new_layer['y']) < 210_000 and min(new_layer['y']) > 90_000:
                    E_layers.append(new_layer)
                elif min(layer['y']) > 140_000:
                    F_layer.append(new_layer)
                else:
                    F_layer.append(new_layer)

    E_layer =  None
    Es_layer = None

    if len(E_layers) == 2:
        if len(E_layers[0]) > len(E_layers[1]):
            E_layer, Es_layer = E_layers[0], E_layers[1]
        else:
            E_layer, Es_layer = E_layers[1], E_layers[0]

    elif len(E_layers) == 1:
        E_layer = E_layers[0]

    elif len(E_layers) != 0:
        E_layer = pd.concat(E_layers)

    # print(F_layer[0])
    # print(sublayer[-1])
    return pd.concat(F_layer), E_layer, Es_layer, pd.DataFrame(sublayer, columns=['x', 'y'])



def separate_layers_second_pass(*layers):
    sublayer = pd.DataFrame(columns=['x', 'y'])
    new_layers = []
    for layer in layers:
        df = pd.DataFrame(layer, columns=['x', 'y'])
        x = df['x'].unique()
        for i in range(2, len(df['x'].unique())-2):
            try:
                sum_x = sum(df.loc[df['x'] == x[i-1]]['y']) + sum(df.loc[df['x'] == x[i+1]]['y']) + sum(df.loc[df['x'] == x[i-2]]['y']) + sum(df.loc[df['x'] == x[i+2]]['y'])
                len_x = len(df.loc[df['x'] == x[i-1]]['y']) + len(df.loc[df['x'] == x[i+1]]['y']) + len(df.loc[df['x'] == x[i-2]]['y']) + len(df.loc[df['x'] == x[i+2]]['y'])
            except TypeError:
                print(df.loc[df['x'] == x[i-1]]['y'], df.loc[df['x'] == x[i+1]])
            average_x = sum_x / len_x
            # print(average_x, df.loc[df['x'] == x[i]])
            # print(df['x'])
            # print(df['y'])
            sublayer = pd.concat([sublayer, df.loc[(df['x'] == x[i]) & ((df['y'] > average_x + 1500 * 18) | (df['y'] < average_x - 1500 * 25))]])
            df.drop(df[(df['x'] == x[i]) & ((df['y'] > average_x + 1500 * 18) | (df['y'] < average_x - 1500 * 25))].index)
        new_layers.append(df)

    return new_layers, sublayer


def separate_layers_second_pass_v2(*layers):
    sublayer = pd.DataFrame(columns=['x', 'y'])
    new_layers = []
    for layer in layers:
        df = pd.DataFrame(layer, columns=['x', 'y'])
        x = df['x'].unique()
        for i in range(1, len(x)):
            for point1 in df.loc[df['x'] == x[i]]['y']:
                # print(point1)
                # print(type(point1))
                dists = []
                for point2 in df.loc[df['x'] == x[i-1]]['y']:
                    dists.append(np.sqrt(((x[i] - x[i-1]) ** 2) + ((point2 - point1) ** 2) / 50) < 1500 * 5)

                if not any(dists):
                    sublayer = pd.concat([sublayer, pd.DataFrame([[x[i], point1]], columns=df.columns)])
                    df.drop(df.loc[(df['x'] == x[i]) & (df['y'] == point1)].index)
        new_layers.append(df)
    return new_layers, sublayer


# def separate_F(df: pd.DataFrame):
#     increase = []
#     last_increase = []
#     max_increase = 0
#     i_max = 0
#     begin = len(df)//15
#     for i, v in enumerate(df["y'"][begin:], start=begin):
#         if v >= 0:
#             increase.append(i)
#
#         if v < 0 and increase:
#             if min(df["y'"][i + 1], df["y'"][i + 2], df["y'"][i + 3]) and max(df["y'"][i + 1], df["y'"][i + 2], df["y'"][i + 3]) <= 0:
#                 if (incr := df["y"][increase[-1]] - df["y"][increase[0]]) > max_increase:
#                     print(incr)
#                     max_increase = incr
#                     i_max = i
#                     last_increase = increase.copy()
#                     print(last_increase)
#                     print(i_max)
#                 increase.clear()
#             else:
#                 increase.append(i)
#
#     print(last_increase)
#     print(i_max)
#     if i_max > len(df)//3 * 2:
#         increase = []
#         max_increase = 0
#         i_max = 0
#         for i, v in enumerate(df["y'"].iloc[begin:last_increase[0]], start=begin):
#             if v >= 0:
#                 increase.append(i)
#
#             if v < 0 and increase:
#                 if min(df["y'"][i + 1], df["y'"][i + 2], df["y'"][i + 3]) and max(df["y'"][i + 1], df["y'"][i + 2], df["y'"][i + 3]) <= 0:
#                     if (incr := df["y"][increase[-1]] - df["y"][increase[0]]) > max_increase:
#                         print(incr)
#                         max_increase = incr
#                         i_max = i
#                         print(i_max)
#                     increase.clear()
#
#     return df.loc[df.index <= i_max], df.loc[df.index > i_max]



def layer_average(layer, quantity=100):
    len_layer = len(layer)
    average_layer = layer[len_layer - quantity if len_layer > quantity else 0:]
    layer_x, layer_y = zip(*average_layer)
    # print(layer_y)
    result = sum(layer_y)/(quantity if len_layer > quantity else len_layer)
    # print(result)
    return result


def layer_max(layer):
    layer_x, layer_y = zip(*layer)
    return max(layer_y)



def separate_F(df_: pd.DataFrame):
    df = median_filter_df(df_)
    target_df = simplification(df)
    target_df = derivative(target_df)
    target_df = derivative(target_df)
    target_df = target_df[len(target_df)//5:-len(target_df)//5]

    target_df.reset_index(inplace=True, drop=True)

    for i, v in enumerate(target_df["y'"]):
        if v >= 0 and i < len(target_df)-2 and target_df.iloc[i+1]["y'"] < 0:
            x1 = target_df.iloc[i]["x"]
            x2 = target_df.iloc[i+2]["x"]
            print("Сильный перегиб")
            max_y = df[df["x"].between(x1, x2)].idxmax()['y']
            return df.iloc[max_y]["x"]

    for i, v in enumerate(target_df["y'"]):
        if (v >= 0 and target_df.iloc[i]["y''"] > 0) and i < len(target_df)-2 and (target_df.iloc[i+1]["y''"] < 0):
            x1 = target_df.iloc[i]["x"]
            x2 = target_df.iloc[i+2]["x"]
            print("Слабый перегиб")
            return (x1+x2)/2

    for i, v in enumerate(target_df["y'"]):
        if (v >= 0 and target_df.iloc[i]["y''"] < 0) and target_df.iloc[i]["y'"] < 0.01:
            x1 = target_df.iloc[i-1]["x"]
            x2 = target_df.iloc[i+1]["x"] if i != len(target_df)-1 else target_df.iloc[i]["x"]
            print("без перегиба")
            return (x1+x2)/2



