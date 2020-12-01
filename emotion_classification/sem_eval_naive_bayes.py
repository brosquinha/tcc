import csv
from functools import reduce
from random import choice

import numpy as np

from dataset.tweet_db import TweetDB
from emotion_classification.datasets.sem_eval import SemEval
from emotion_classification.naive_bayes import NaiveBayes


class SemEvalNaiveBayes(NaiveBayes):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.semeval = SemEval(*args, **kwargs)
        self.all_data = self.semeval.all_data
        self.emotion_name = self.semeval.emotion_name
        self.emotions = self.semeval.emotions
        self._get_text_from_data = self.semeval.get_x_data
        self._get_y_data = self.semeval.get_y_data


class AllEmotionsSemEvalNaiveBayes(SemEvalNaiveBayes):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emotions = ['anger', 'fear', 'joy', 'sadness']
        self.chart_name = 'all_emotions.png'
    
    def train_model(self):
        for model in self._repeated_k_fold_training(self.all_data, self._get_text_from_data, self._get_y_data):
            self.test_model(model)
        self.logger.info("Accuracies mean for all emotions: %.2f%%" % (np.array(self.accuracy).mean() * 100))
        return self.text_clf

    def test_model(self, model):
        results = model.predict_proba(self._get_text_from_data(self.test_data))
        correct_count = 0
        total_emotions = len(results) * len(self.emotions)
        predicted = []
        y_data = self._get_y_data(self.test_data)
        y_classes = list(sorted(set(self.train_data_y)))

        for result_index, result in enumerate(results):
            expected = y_data[result_index]
            max_prob = max(result)
            result_y = y_classes[list(result).index(max_prob)]
            predicted.append(result_y)
            result_emotions = result_y
            expected_emotions = expected

            error_length = len(set(expected_emotions).symmetric_difference(set(result_emotions)))
            correct_count += len(self.emotions) - error_length

        self.logger.debug("All emotions model")
        self._get_tests_metrics(correct_count, total_emotions)
        self._plot_confusion_matrix(
            predicted,
            y_data,
            labels=(list(set(y_data+predicted))),
            filename=self.chart_name,
            largesize=True
        )


class AllSingleEmotionsSemEvalNaiveBayes(AllEmotionsSemEvalNaiveBayes):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.all_data = reduce(self._transform_multiple_emotions_into_one, self.all_data, [])
        np.random.shuffle(self.all_data)
        self.chart_name = 'all_single_emotions.png'

    def test_model(self, model):
        unique_test_data = np.unique(self._get_text_from_data(self.test_data))
        results = model.predict_proba(unique_test_data)
        correct_count = 0
        total = len(results)
        predicted = []
        y_data = []
        y_classes = list(sorted(set(self.train_data_y)))

        for result_index, result in enumerate(results):
            result_text = unique_test_data[result_index]
            expected = self._get_y_data([x for x in self.test_data if self._get_text_from_data([x])[0] == result_text])
            max_prob = max(result)
            result_y = y_classes[list(result).index(max_prob)]
            predicted.append(result_y)

            if result_y in expected:
                correct_count += 1
                y_data.append(result_y)
            else:
                y_data.append(choice(expected))

        self.logger.debug("All single emotions model")
        self._get_tests_metrics(correct_count, total)
        self._plot_confusion_matrix(
            predicted,
            y_data,
            labels=self.emotions,
            filename=self.chart_name,
            largesize=False
        )

    def _transform_multiple_emotions_into_one(self, acc, data):
        multi_result = ''.join([data[2], data[5], data[6], data[10]])
        for i, x in enumerate(multi_result):
            if x == '0':
                continue
            acc.append([data[0], data[1], self.emotions[i]])
        return acc

    def _get_y_data(self, data):
        return [x[2] for x in data]


class SingleEmotionSemEvalNaiveBayes(SemEvalNaiveBayes):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emotion_index = self.emotions.index(self.emotion_name)
        self.emotion_label = self.emotion_name

    def train_model(self):
        for model in self._repeated_k_fold_training(self.all_data, self._get_text_from_data, self._get_y_data):
            self.test_model(model)
        self.logger.info("Accuracies mean for %s: %.2f%%" % (self.emotion_label, (np.array(self.accuracy).mean() * 100)))
        return self.text_clf

    def test_model(self, model):
        results = model.predict_proba(self._get_text_from_data(self.test_data))
        correct_count = 0
        total_emotion = len(results)
        result_index = 0
        predicted = []
        y_classes = list(sorted(set(self.train_data_y)))
        y_test_data = self._get_y_data(self.test_data)
        
        for result in results:
            expected = int(y_test_data[result_index])
            max_prob = max(result)
            result_y = y_classes[list(result).index(max_prob)]
            predicted.append(result_y)

            if expected == int(result_y):
                correct_count += 1
            result_index += 1

        self.logger.debug("Single emotion model for %s" % self.emotion_label)
        self._get_tests_metrics(correct_count, total_emotion)
        self._plot_confusion_matrix(
            predicted,
            y_test_data,
            largesize=False,
            labels=['No', 'Yes'],
            filename='{}.png'.format(self.emotion_label),
            title='Confusion matrix for %s (SemEval dataset)' % self.emotion_label
        )

    def classify_replies_tweets(self):
        tdb = TweetDB()
        tweets = list(tdb.all_replies())
        tweets_with_emotion_indexes, total = self.classify_sentences([t.text for t in tweets])
        tweets_with_emotion = [int(tweets[i].parent_tweet.retweet_count) for i in tweets_with_emotion_indexes]
        tweets_with_emotion_mean = np.array(tweets_with_emotion).mean() if tweets_with_emotion else 0
        self.logger.info('%d of %d on %s category (retweets average: %.2f)' % (
            len(tweets_with_emotion), total, self.emotion_label, tweets_with_emotion_mean))

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

