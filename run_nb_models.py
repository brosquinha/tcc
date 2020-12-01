import os
from argparse import ArgumentParser

import numpy as np

from emotion_classification.evaluate_tweets_replies import predict_replies
from emotion_classification.datasets.manual_input import ManualInput
from emotion_classification.sem_eval_naive_bayes import (
    AllEmotionsSemEvalNaiveBayes, AllSingleEmotionsSemEvalNaiveBayes,
    SemEvalNaiveBayes, SingleEmotionSemEvalNaiveBayes)
from emotion_classification.tec_naive_bayes import (SingleEmotionTECNaiveBayes,
                                                    TECNaiveBayes)
from emotion_classification.text_emotion_naive_bayes import SingleEmotionTextEmotionNaiveBayes, TextEmotionNaiveBayes
from dataset.tweet_db import TweetDB


def train_semeval_all_emotions_model(*args, **kwargs):
    emotions = SemEvalNaiveBayes().emotions
    nb = AllEmotionsSemEvalNaiveBayes(
        filename='output/all_emotions.pickle',
        ignore_pickle=True,
        log_level=kwargs.get('debug'),
        balance_data=not kwargs.get('no_balance_data')
    )
    nb.train_model()
    nb.save_model()

def train_semeval_single_emotion_models(*args, **kwargs):
    emotions = SemEvalNaiveBayes().emotions
    for emotion in emotions:
        nb = SingleEmotionSemEvalNaiveBayes(
            emotion_name=emotion,
            filename=os.path.join('output', '%s.pickle' % emotion),
            ignore_pickle=True,
            log_level=kwargs.get('debug'),
            balance_data=not kwargs.get('no_balance_data'),
            balance_tolerance=kwargs.get('balance_tolerance')
        )
        nb.train_model()
        nb.save_model()

def train_semeval_all_single_emotion_model(*args, **kwargs):
    nb = AllSingleEmotionsSemEvalNaiveBayes(
        filename='output/all_single_emotions.pickle',
        ignore_pickle=True,
        log_level=kwargs.get('debug'),
        balance_data=not kwargs.get('no_balance_data')
    )
    nb.train_model()
    nb.save_model()

def train_tec_model(*args, **kwargs):
    model = TECNaiveBayes(
        filename='output/tec.pickle',
        ignore_pickle=True,
        log_level=kwargs.get('debug'),
        balance_data=not kwargs.get('no_balance_data')
    )
    model.train_model()
    model.save_model()

def train_single_emotion_tec_model(*args, **kwargs):
    all_emotions = ['surprise', 'anger', 'joy', 'fear', 'disgust', 'sadness']
    for emotion in all_emotions:
        model = SingleEmotionTECNaiveBayes(
            emotion_name=emotion,
            filename='output/%s-tec.pickle' % emotion,
            ignore_pickle=True,
            log_level=kwargs.get('debug'),
            k_splits=kwargs.get('k_splits'),
            balance_data=not kwargs.get('no_balance_data'),
            balance_tolerance=kwargs.get('balance_tolerance')
        )
        model.train_model()
        model.save_model()

def train_text_emotion_model(*args, **kwargs):
    model = TextEmotionNaiveBayes(
        filename='output/text_emotion.pickle',
        ignore_pickle=True,
        log_level=kwargs.get('debug'),
        balance_data=not kwargs.get('no_balance_data'),
        balance_tolerance=kwargs.get('balance_tolerance')
    )
    model.train_model()
    model.save_model()

def train_single_emotion_text_emotion_model(*args, **kwargs):
    all_emotions = ['anger', 'fear', 'joy', 'neutral', 'sadness']
    for emotion in all_emotions:
        model = SingleEmotionTextEmotionNaiveBayes(
            emotion_name=emotion,
            filename='output/%s-text-emotion.pickle' % emotion,
            ignore_pickle=True,
            log_level=kwargs.get('debug'),
            k_splits=kwargs.get('k_splits'),
            balance_data=not kwargs.get('no_balance_data'),
            balance_tolerance=kwargs.get('balance_tolerance')
        )
        model.train_model()
        model.save_model()

def test_tec_with_semeval(*args, **kwargs):
    all_emotions = ['surprise', 'anger', 'joy', 'fear', 'disgust', 'sadness']
    for emotion in all_emotions:
        model = SingleEmotionTECNaiveBayes(
            emotion_name=emotion,
            filename='output/%s-tec.pickle' % emotion,
            ignore_pickle=False,
            log_level=kwargs.get('debug'),
        )
        emotions = SemEvalNaiveBayes().emotions
        semeval = SingleEmotionSemEvalNaiveBayes(
            emotion_name=emotion,
            filename='output/%s.pickle' % emotion
        )
        semeval.test_data = semeval.all_data
        semeval.train_data_y = [0, 1]
        semeval.logger.info('Testing for %s' % emotion)
        semeval.test_model(model.text_clf)

def test_semeval_with_tec(*args, **kwargs):
    all_emotions = ['surprise', 'anger', 'joy', 'fear', 'disgust', 'sadness']
    emotions = SemEvalNaiveBayes().emotions
    for emotion in all_emotions:
        model = SingleEmotionSemEvalNaiveBayes(
            emotion_name=emotion,
            filename='output/%s.pickle' % emotion,
            ignore_pickle=False,
            log_level=kwargs.get('debug')
        )
        tec = SingleEmotionTECNaiveBayes(
            emotion_name=emotion,
            filename='output/%s-tec.pickle' % emotion
        )
        tec.test_data = tec.all_data
        tec.train_data_y = ['0', '1']
        tec.logger.info('Testing for %s' % emotion)
        tec.test_model(model.text_clf)

def test_model_with_manual_input(*args, **kwargs):
    all_emotions = ManualInput().emotions_labels
    if kwargs['dataset'] == 'semeval':
        class_name = SingleEmotionSemEvalNaiveBayes
        pickle_filename = 'output/{}.pickle'
    else:
        class_name = SingleEmotionTECNaiveBayes
        pickle_filename = 'output/{}-tec.pickle'

    for emotion in all_emotions:
        model = class_name(
            emotion_name=emotion,
            filename=pickle_filename.format(emotion),
            ignore_pickle=False,
            log_level=kwargs.get('debug')
        )
        manual_input = ManualInput(emotion_name=emotion)
        model._get_x_data = manual_input.get_x_data
        model._get_text_data = manual_input.get_x_data
        model._get_y_data = manual_input.get_y_data
        model.test_data = manual_input.all_data
        model.train_data_y = ['0', '1']
        model.test_model(model.text_clf)

def test_tec_single_emotion_with_replies(*args, **kwargs):
    emotions, tweets_data = _classify_replies_with_emojis(prefix=':: ')

    for emotion in emotions:
        print(emotion)
        nb = SingleEmotionTECNaiveBayes(
            emotion_name=emotion,
            ignore_pickle=False,
            filename=os.path.join('output', '%s-tec.pickle' % emotion),
            log_level=kwargs.get('debug')
        )
        nb.test_data = tweets_data
        nb.train_data_y = ['0', '1']
        nb.test_model(nb.text_clf)

def test_tec_all_emotions_with_replies(*args, **kwargs):
    emotions, tweets_data = _classify_replies_with_emojis(prefix=':: ')

    nb = TECNaiveBayes(
        filename='output/tec.pickle',
        ignore_pickle=False,
        log_level=kwargs.get('debug')
    )
    nb.test_data = tweets_data
    nb.train_data_y = ['surprise', 'anger', 'joy', 'fear', 'disgust', 'sadness']
    nb.test_model(nb.text_clf)

def test_semeval_with_replies(*args, **kwargs):
    emotions, tweets_data = _classify_replies_with_emojis()

    nb = AllSingleEmotionsSemEvalNaiveBayes(
        filename='output/all_single_emotions.pickle',
        ignore_pickle=False,
        log_level=kwargs.get('debug')
    )
    nb.test_data = tweets_data
    nb.train_data_y = emotions
    nb.test_model(nb.text_clf)

def train_multiple_datasets(*args, **kwargs):
    from sklearn.model_selection import train_test_split
    
    from emotion_classification.datasets.aggregate import Aggregate
    from emotion_classification.datasets.sem_eval import SemEval
    from emotion_classification.datasets.tec import TEC
    from emotion_classification.datasets.text_emotion import TextEmotion

    all_emotions = ['anger', 'fear', 'joy', 'sadness']
    for emotion in all_emotions:
        model = SingleEmotionTECNaiveBayes(
            emotion_name=emotion,
            filename='output/%s-multi-datasets.pickle' % emotion,
            ignore_pickle=True,
            log_level=kwargs.get('debug'),
            k_splits=kwargs.get('k_splits'),
            balance_data=not kwargs.get('no_balance_data'),
            balance_tolerance=kwargs.get('balance_tolerance')
        )
        aggregate = Aggregate(SemEval(emotion_name=emotion), TEC(emotion_name=emotion))
        model.logger.info(f'Dataset length: {len(aggregate.all_data)}')
        all_data, test_data = train_test_split(aggregate.all_data, test_size=0.2)
        model.all_data = all_data
        model._get_x_data = aggregate.get_x_data
        model._get_y_data = aggregate.get_y_data
        model.train_model()

        model.test_data = test_data
        model.test_model(model.text_clf)
        
        model.save_model()
    

def _classify_replies_with_emojis(prefix=''):
    emotions = ['anger', 'fear', 'joy', 'sadness']
    emojis = [
        ['😠', '😡', '😤', '🤬'], #anger
        ['😰', '😱', '😨', '😟'], #fear
        ['😂', '😁', '😄', '😊'], #, '🤣', '😀', '😃', '😄', '😆', '😍', '😋'], #joy
        ['💔', '😢', '😭', '😔'] #sadness
    ]
    
    tdb = TweetDB()
    tweets_data = []
    for i, emoji_list in enumerate(emojis):
        tweets_with_an_emoji = []
        other_emojis = [e for el in emojis for e in el if e not in emoji_list]
        for emoji in emoji_list:
            tweets_list = filter(
                lambda t: None if any(True if e in t.text else False for e in other_emojis) else True,
                list(tdb.all_replies_like(emoji))
            )
            tweets_with_an_emoji += tweets_list
        tweets_with_emoji = list(set(tweets_with_an_emoji))
        tweets_data += list(map(lambda t: ['', t.text, f'{prefix}{emotions[i]}'], tweets_with_emoji))
    
    return emotions, tweets_data

def main(*args, **kwargs):
    functions = {
        'dataset': {
            'tec': {
                'action': {
                    'train': {
                        'model': {
                            'all-emotions': train_tec_model,
                            'single-emotion': train_single_emotion_tec_model,
                            'all-single-emotions': lambda *args, **kwargs: print('Invalid model option')
                        }
                    },
                    'validate': test_model_with_manual_input,
                    'test': test_tec_with_semeval,
                    'tweets': {
                        'model': {
                            'all-emotions': test_tec_all_emotions_with_replies,
                            'single-emotion': test_tec_single_emotion_with_replies
                        }
                    }
                },
            },
            'semeval': {
                'action': {
                    'train': {
                        'model': {
                            'all-emotions': train_semeval_all_emotions_model,
                            'single-emotion': train_semeval_single_emotion_models,
                            'all-single-emotions': train_semeval_all_single_emotion_model
                        }
                    },
                    'tweets': {
                        'model': {
                            'all-single-emotions': test_semeval_with_replies,
                            'single-emotion': predict_replies,
                            'all-emotions': train_multiple_datasets
                        }
                    },
                    'validate': test_model_with_manual_input,
                    'test': test_semeval_with_tec
                }
            },
            'text-emotion': {
                'action': {
                    'train': {
                        'model': {
                            'all-emotions': train_text_emotion_model,
                            'single-emotion': train_single_emotion_text_emotion_model
                        }
                    }
                }
            }
        }
    }

    decider = functions['dataset'][kwargs['dataset']]['action'][kwargs['action']]
    while not callable(decider):
        decider_name = list(decider.keys())[0]
        value = decider[decider_name][kwargs[decider_name]]
        decider = value
    decider(*args, **kwargs)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['semeval', 'tec', 'text-emotion'], default='semeval', help='Dataset to be used')
    parser.add_argument('--action', choices=['train', 'tweets', 'test', 'validate'], default='train', help='Whether to train models or predict replies')
    parser.add_argument('--model', choices=['all-emotions', 'single-emotion', 'all-single-emotions'], default='single-emotion', help='Whether to use all emotions or single emotions models')
    parser.add_argument('--filepath', type=str, help='Model pickle filepath for replies emotion prediction')
    parser.add_argument('--k-splits', type=int, default=4, help='Number of K splits for cross-validation')
    parser.add_argument('--debug', action='store_const', const='debug', default='info', help='Enables debug logging')
    parser.add_argument('--no-balance-data', action='store_const', const=True, default=False, help='Disables dataset balacing via undersampling')
    parser.add_argument('--balance-tolerance', type=float, default=1.0, help='Amount to be allowed to each group to excess')
    args = parser.parse_args()

    main(**vars(args))
