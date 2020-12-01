import csv

import numpy as np

class TEC():

    def __init__(self, emotion_name='tec', *args, **kwargs):
        self.emotion_name = emotion_name
        self.all_data = self._read_tec_csv_file()
        self.emotions_labels = ['surprise', 'anger', 'joy', 'fear', 'disgust', 'sadness']
    
    def get_x_data(self, data):
        return [s[1] for s in data]

    def get_y_data(self, data):
        if self.emotion_name == 'tec':
            return [x[2].replace(":: ", "") for x in data]
        return ['1' if x[2] == ':: {}'.format(self.emotion_name) else '0' for x in data]

    def normalize_all_data_for_emotion(self):
        self.all_data = [x[:2] + (['1'] if x[2] == ':: {}'.format(self.emotion_name) else ['0']) for x in self.all_data]
    
    def _read_tec_csv_file(self):
        with open("dataset/source/TEC.csv") as f:
            content = csv.reader(f, delimiter='\t')
            data = list(content)
        return data
