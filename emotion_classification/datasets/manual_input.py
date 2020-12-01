import csv

import numpy as np

class ManualInput():

    def __init__(self, emotion_name='tec', *args, **kwargs):
        self.emotion_name = emotion_name
        self.all_data = self._read_csv_file()[1:]
        self.all_data = [x for x in self.all_data if x[3]]
        self.emotions_labels = ['anger', 'fear', 'joy', 'sadness']
    
    def get_x_data(self, data):
        return [s[1] for s in data]

    def get_y_data(self, data):
        return ['1' if x[3 + self.emotions_labels.index(self.emotion_name)] == 'TRUE' else '0' for x in data]

    def normalize_all_data_for_emotion(self):
        self.all_data = [x[:3] + (['1'] if x[3 + self.emotions_labels.index(self.emotion_name)] == 'TRUE' else ['0']) for x in self.all_data]
    
    def _read_csv_file(self):
        with open("dataset/source/respostas_classificadas_manualmente.csv") as f:
            content = csv.reader(f)
            data = list(content)
        return data
