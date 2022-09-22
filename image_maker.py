import json

import matplotlib.pyplot as plt
import numpy as np


def json_to_png_for_describe(path: str):
    data_list = []

    with open(path) as f:
        json_obj = json.load(f)

    row_label_array = np.array(list(json_obj.keys()))
    col_label_array = np.array(list(json_obj['bpm'].keys()))

    i = 0
    for colLabel, row in json_obj.items():
        data_list += [list(row.values())]

    data_array = np.array(data_list)

    fig = plt.figure(figsize=(80, 60), dpi=100)
    plt.axis('off')
    # 表を描写
    table = plt.table(
        cellText=np.round(data_array, 2),
        rowLabels=row_label_array,
        colLabels=col_label_array,
        cellLoc='center',
        loc='center')  # デフォルトはグラフの下に表示なので、centerを指定して中央に設定

    table.set_fontsize(100)
    table.scale(1, 15)
    fig.savefig('describe.png', dpi=fig.dpi)


def json_to_png_for_evaluation(path: str):
    with open(path) as f:
        json_obj = json.load(f)

    col_label_array = np.array(list(json_obj.keys()))
    data_array = np.array([list(json_obj.values())])

    fig = plt.figure(figsize=(10, 5), dpi=300)
    plt.axis('off')
    table = plt.table(
        cellText=np.round(data_array, 2),
        colLabels=col_label_array,
        cellLoc='center',
        loc='center')

    table.set_fontsize(100)
    table.scale(1, 5)
    fig.savefig('modelEvaluation.png', dpi=fig.dpi)
    
# json_to_png_for_describe('describe.json')
# json_to_png_for_evaluation('modelEvaluation.json')