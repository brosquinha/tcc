import csv


class TextEmotion():
    
    emotion_labels = {
        'empty': 'neutral',
        'sadness': 'sadness',
        'enthusiasm': 'joy',
        'neutral': 'neutral',
        'worry': 'fear',
        'surprise': 'neutral',
        'love': 'joy',
        'fun': 'joy',
        'hate': 'anger',
        'happiness': 'joy',
        'boredom': 'neutral',
        'relief': 'joy',
        'anger': 'anger'
    }
    
    def __init__(self, emotion_name='all', *args, **kwargs):
        self.emotion_name = emotion_name
        self.all_data = self._read_csv_file()
        self.emotions_labels = ['anger', 'fear', 'joy', 'neutral', 'sadness']

    def get_x_data(self, data):
        return [s[3] for s in data]

    def get_y_data(self, data):
        if self.emotion_name == 'all':
            return [self.emotion_labels[s[1]] for s in data]
        return ['1' if self.emotion_labels[s[1]] == self.emotion_name else '0' for s in data]

    def normalize_all_data_for_emotion(self):
        self.all_data = [x[:1] + (['1'] if self.emotion_name == self.emotion_labels[x[1]] else ['0']) + x[2:] for x in self.all_data]

    def _read_csv_file(self):
        with open('dataset/source/text_emotion.csv') as f:
            content = csv.reader(f, delimiter=',')
            data = list(content)
        return data[1:]
