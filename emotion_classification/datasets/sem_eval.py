import csv

import numpy as np

class SemEval():
    
    def __init__(self, emotion_name='semeval', *args, **kwargs):
        self.emotion_name = emotion_name
        data = self._read_semeval_csv_file('dataset/source/2018-E-c-En-train.txt')
        self.emotions = data[0][2:]
        self.all_data = data[1:] + self._read_semeval_csv_file('dataset/source/2018-E-c-En-test-gold.txt')[1:]
        self.all_data += self._read_semeval_csv_file('dataset/source/2018-E-c-En-dev.txt')[1:]

    def get_x_data(self, data):
        return [s[1] for s in data]
    
    def get_y_data(self, data):
        if self.emotion_name == 'semeval':
            return [''.join([x[2], x[5], x[6], x[10]]) for x in data]
        return [int(x[self.emotions.index(self.emotion_name) + 2]) for x in data]

    def _read_semeval_csv_file(self, filename):
        with open(filename, encoding="utf-8") as f:
            content = csv.reader(f, delimiter='\t')
            data = list(content)
        return data
