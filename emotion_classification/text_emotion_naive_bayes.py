import csv

import numpy as np

from emotion_classification.datasets.text_emotion import TextEmotion
from emotion_classification.naive_bayes import NaiveBayes


class TextEmotionNaiveBayes(NaiveBayes):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tec = TextEmotion(*args, **kwargs)
        self.emotion_name = self.tec.emotion_name
        self.all_data = self.tec.all_data
        self.emotions_labels = self.tec.emotions_labels
        self._get_x_data = self.tec.get_x_data
        self._get_y_data = self.tec.get_y_data

    def train_model(self):
        for model in self._repeated_k_fold_training(self.all_data, self._get_x_data, self._get_y_data):
            self.test_model(model)
        self.logger.info("Accuracies mean for %s: %.2f%%" % (self.emotion_name, (np.array(self.accuracy).mean() * 100)))
        return self.text_clf

    def test_model(self, model):
        self.logger.debug('Testing...')
        results = model.predict_proba(self._get_x_data(self.test_data))
        correct_count = 0
        total_emotion = len(results)
        predicted = []
        y_test_data = self._get_y_data(self.test_data)
        y_classes = list(sorted(set(self.train_data_y)))
        self.logger.debug('Analizing test results...')
        for result_index, result in enumerate(results):
            expected = y_test_data[result_index]
            max_prob = max(result)
            result_y = y_classes[list(result).index(max_prob)]
            predicted.append(result_y)

            if expected == result_y:
                correct_count += 1

        self.logger.debug("All emotions model for Text Emotion dataset %s" % self.emotion_name)
        self._get_tests_metrics(correct_count, total_emotion)
        self._plot_confusion_matrix(
            predicted,
            y_test_data,
            labels=self.emotions_labels,
            filename='%s-text-emotion.png' % self.emotion_name,
            title='Confusion matrix for %s (Text Emotion dataset)' % self.emotion_name
        )


class SingleEmotionTextEmotionNaiveBayes(TextEmotionNaiveBayes):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emotions_labels = ['0', '1']

    def classify_sentences(self, sentences):
        results = self.text_clf.predict_proba(sentences)
        positive_tweets = []
        y_classes = list(sorted(['0', '1']))

        for index, result in enumerate(results):
            max_prob = max(result)
            result_y = y_classes[list(result).index(max_prob)]
            if int(result_y):
                positive_tweets.append(index)
                
        return positive_tweets, len(sentences)
