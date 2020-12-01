import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from emotion_classification.base import Base


class NaiveBayes(Base):
    
    def __init__(self, filename='model.pickle', ignore_pickle=False, log_level='info', **kwargs):
        super().__init__(log_level=log_level, **kwargs)
        self.filename = filename
        self.accuracy = []

        try:
            with open(self.filename, 'rb') as f:
                self.text_clf: Pipeline = pickle.load(f)
        except:
            self.text_clf: Pipeline = None
        if ignore_pickle:
            self.text_clf: Pipeline = None

    def save_model(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.text_clf, f)

    def train_model(self, *args, **kwargs):
        raise NotImplementedError

    def test_model(self, *args, **kwargs):
        raise NotImplementedError

    def _train_model(self, x_data, y_data):
        if not self.text_clf:
            self.text_clf = Pipeline([('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                ('clf', MultinomialNB())], verbose=self.logger.level < 20)

        self.train_data_y = np.array(y_data)

        self.text_clf.fit(np.array(x_data), self.train_data_y)

    def _get_tests_metrics(self, correct_count, total_emotion):
        self.logger.debug('Got %d correct out of %d' % (correct_count, total_emotion))
        self.logger.debug("That's %.2f%% precision rate" % ((correct_count / total_emotion) * 100))
        self.accuracy.append((correct_count / total_emotion))
        self.logger.info("Accuracies mean: %.2f%%" % (np.array(self.accuracy).mean() * 100))
