
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


def plot_custom_confusion_matrix(data, emotion_name, dataset_name, filename_format=None):
    maxValue = max([max(data[0]), max(data[1])])

    df_cm = pd.DataFrame(data, range(2), range(2))
    sn.set(font_scale=1) # for label size

    # Create the heatmap with the confusion matrix values
    ax = sn.heatmap(df_cm, annot=True, cmap='Blues', fmt="d")

    # Repaint all annotation values that are less than (0.5 * maxValue) to blue color
    # It's not pretty, but it's the best thing I got to make work
    # We're basically remaking the heatmap, but only recalculating the cells that are off the mask
    #   without the colorbar (the bar on the side), and alpha = 0 makes this new heatmap cells invisible (except for the annotation values)
    ax = sn.heatmap(df_cm, annot=True, cmap='Blues', alpha = 0.0, fmt="d", mask= np.array(data) > 0.5 * maxValue, ax=ax, annot_kws= {'color': 'b'}, cbar = False)

    # Set plot title and axis labels
    plt.title(f'{dataset_name} LSTM {emotion_name}')
    
    plt.xlabel('Predicted result')
    ax.set_xticklabels(["No", "Yes"]) 

    plt.ylabel('Expected result')
    ax.set_yticklabels(["No", "Yes"], rotation=0)

    # Draw the plot outline
    corr = df_cm.corr()
    ax.axhline(y=0, color='k',linewidth=1)
    ax.axhline(y=corr.shape[1], color='k',linewidth=2)
    ax.axvline(x=0, color='k',linewidth=1)
    ax.axvline(x=corr.shape[0], color='k',linewidth=2)

    if not filename_format:
        return plt
    plt.savefig(filename_format.format(emotion_name=emotion_name, dataset_name=dataset_name))

if __name__ == "__main__":
    dados = [[6022, 1001], [3489, 471]]
    emotion = "nome-teste"
    dataset = "dataset-teste"
    filename_format = 'output/{emotion_name}-{dataset_name}-lstm.png'

    plot_custom_confusion_matrix(
        data=dados, emotion_name=emotion, dataset_name=dataset, filename_format=filename_format)
