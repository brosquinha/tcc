from argparse import ArgumentParser

import numpy as np
from sklearn.model_selection import train_test_split

from emotion_classification.lstm import LSTMBase
from emotion_classification.datasets.aggregate import Aggregate
from emotion_classification.datasets.manual_input import ManualInput
from emotion_classification.datasets.sem_eval import SemEval
from emotion_classification.datasets.tec import TEC
from emotion_classification.datasets.text_emotion import TextEmotion

semeval_emotions = {
    'anger': 2,
    'sadness': 10,
    'fear': 5,
    'joy': 6
}

def train(*args, **kwargs):
    emotion_name = kwargs['emotion_name']
    if not emotion_name:
        raise Exception("Invalid emotion")
    
    k_splits = int(kwargs.get('k_splits'))
    lstm = LSTMBase(
        emotion_name=emotion_name,
        dataset_name='-'.join(kwargs['dataset']),
        load_h5=False,
        log_level=kwargs.get('debug'),
        balance_data=not kwargs.get('no_balance_data'),
        k_splits=k_splits
    )
    lstm.x_function = lambda data: data[1]
    dataset, dataset_x_function, dataset_y_function = get_dataset_data(
        kwargs['dataset'], lstm, emotion_name)
    
    if k_splits > 1:
        model = lstm.train_with_cross_validation(dataset.all_data, dataset_x_function, dataset_y_function)
        x_test, y_test = lstm.prepare_data(lstm.test_data)
        lstm.test_model(model, x_test, y_test)
    else:
        x_data, y_data = lstm.prepare_data(dataset.all_data)
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
        model = lstm.train_model(x_train, y_train)
        lstm.test_model(model, x_test, y_test)

    lstm.save_model()

    lstm.plot_confusion_matrix(model, x_test, y_test)
    print('Above result for {}'.format(emotion_name))

def test(*args, **kwargs):
    emotion_name = kwargs['emotion_name']
    if not emotion_name:
        raise Exception("Invalid emotion")
    
    k_splits = int(kwargs.get('k_splits'))
    lstm = LSTMBase(
        emotion_name=emotion_name,
        dataset_name='-'.join(kwargs['dataset']),
        load_h5=True,
        log_level=kwargs.get('debug')
    )
    
    if kwargs['action'] == 'validate':
        other_dataset = 'manual-input'
    else:
        other_dataset = 'tec' if 'semeval' in kwargs['dataset'] else 'semeval'
    
    lstm.x_function = lambda data: data[1]

    dataset, _, _ = get_dataset_data(other_dataset, lstm, emotion_name)
    x_test, y_test = lstm.prepare_data(dataset.all_data)

    lstm.test_model(lstm.model, x_test, y_test)

    lstm.plot_confusion_matrix(lstm.model, x_test, y_test)
    print('Above result for {}'.format(emotion_name))

def get_dataset_data(dataset, lstm, emotion_name):
    datasets_instances = []
    if 'manual-input' in dataset:
        datasets_instances.append(ManualInput(emotion_name=emotion_name))
    if 'semeval' in dataset:
        datasets_instances.append(SemEval(emotion_name=emotion_name))
    if 'text-emotion' in dataset:
        datasets_instances.append(TextEmotion(emotion_name=emotion_name))
    if 'tec' in dataset:
        datasets_instances.append(TEC(emotion_name=emotion_name))
    lstm.x_function = lambda data: data[0]
    lstm.y_function = lambda data: data[1]
    aggregate = Aggregate(*datasets_instances)
    return aggregate, aggregate.get_x_data, aggregate.get_y_data

def main(*args, **kwargs):
    action = kwargs['action']
    if action == 'train':
        train(*args, **kwargs)
    else:
        test(*args, **kwargs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['semeval', 'tec', 'text-emotion'], nargs='+', help='Dataset to be used')
    parser.add_argument('--action', choices=['train', 'test', 'validate'], default='train', help='Whether to train or test model')
    parser.add_argument('--emotion-name', choices=list(semeval_emotions.keys()) + ['tec'], help='Emotion to be used')
    parser.add_argument('--k-splits', type=int, default=1, help='Number of K splits for cross-validation')
    parser.add_argument('--no-balance-data', action='store_const', const=True, default=False, help='Disables dataset balacing via undersampling')
    parser.add_argument('--debug', action='store_const', const='debug', default='info', help='Enables debug logging')
    args = parser.parse_args()

    main(**vars(args))
    
