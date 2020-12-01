import os
from argparse import ArgumentParser

from emotion_classification.evaluate_tweets_replies import predict_top_retweeted_fake_news_tweets, plot_top_retweeted_tweets_analisys
from emotion_classification.lstm import LSTMBase
from emotion_classification.sem_eval_naive_bayes import SingleEmotionSemEvalNaiveBayes
from emotion_classification.tec_naive_bayes import SingleEmotionTECNaiveBayes
from emotion_classification.text_emotion_naive_bayes import SingleEmotionTextEmotionNaiveBayes


class SemEvalLSTM(LSTMBase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(dataset_name='semeval', log_level='info', *args, **kwargs)


class TECLSTM(LSTMBase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(dataset_name='tec', log_level='info', *args, **kwargs)


class TextEmotionLSTM(LSTMBase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(dataset_name='text-emotion', log_level='info', *args, **kwargs)


def main(datasets, techs, consolidate, *args, **kwargs):
    models = []
    
    if 'semeval' in datasets:
        if 'naive-bayes' in techs:
            models.append((SingleEmotionSemEvalNaiveBayes, os.path.join('output', '{}.pickle')))
        if 'lstm' in techs:
            models.append((SemEvalLSTM, os.path.join('output', '{}-semeval-lstm.h5')))
    if 'tec' in datasets:
        if 'naive-bayes' in techs:
            models.append((SingleEmotionTECNaiveBayes, os.path.join('output', '{}-tec.pickle')))
        if 'lstm' in techs:
            models.append((TECLSTM, os.path.join('output', '{}-tec-lstm.h5')))
    if 'text-emotion' in datasets:
        if 'naive-bayes' in techs:
            models.append((SingleEmotionTextEmotionNaiveBayes, os.path.join('output', '{}-text-emotion.pickle')))
        if 'lstm' in techs:
            models.append((TextEmotionLSTM, os.path.join('output', '{}-text-emotion-lstm.h5')))

    try:
        predict_top_retweeted_fake_news_tweets(models, consolidate)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    finally:
        print('Classification stoped. Plotting charts...')
        for tech in techs:
            for dataset in datasets:
                plot_top_retweeted_tweets_analisys((dataset, tech))
                print(f'{dataset}-{tech} plotted')
            if consolidate:
                plot_top_retweeted_tweets_analisys(('consolidation', tech))
                print(f'consolidation-{tech} plotted')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--datasets', choices=['semeval', 'tec', 'text-emotion'], nargs='+', help='Dataset to be used')
    parser.add_argument('--techs', choices=['naive-bayes', 'lstm'], nargs='+', help='Machine learning model to be used')
    parser.add_argument('--consolidate', type=int, choices=range(1, 4), default=2, help='Number of models needed to confirm a positive emotion classification')
    args = parser.parse_args()

    main(**vars(args))
