import os

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import LsiModel, Word2Vec
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import (LSTM, Bidirectional, Conv1D, Dense, Dropout,
                          Embedding, Flatten, GlobalAveragePooling1D,
                          GlobalMaxPooling1D, Input, MaxPool1D, concatenate)
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import L1L2
from keras.utils import plot_model

from emotion_classification import nn_models
from emotion_classification.base import Base
from emotion_classification.metrics_evaluation.f1_metric import F1Score


class LSTMBase(Base):
    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000

    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 140

    # This is fixed.
    EMBEDDING_DIM = 200

    # The next layer is the LSTM layer with N memory units.
    LSTM_LAYER_MEMORY_UNITS = 100

    # The output layer must create M output values, one for each class.
    OUTPUT_LAYER_VALUES = 2

    epochs = 50
    batch_size = 32
    dropout = 0.2

    def __init__(
            self, emotion_name: str, dataset_name: str, load_h5=True, f1_score=False,
            word2vec=False, lsi=False, loss='binary_crossentropy', use_tests=False,
            architecture='lstm_cnn_bi', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.architecture = architecture
        self.x_function = None
        self.y_function = None
        self.test_data = []
        self.emotion_name = emotion_name
        self.dataset_name = dataset_name
        self.use_word2vec = word2vec
        self.use_lsi = lsi
        self.keras_loss = loss
        self.use_f1_score = f1_score
        self.use_tests = use_tests
        try:
            self.model = load_model(self._get_model_filename())
        except:
            self.model: Sequential = None
        if not load_h5:
            self.model = None
    
    def train_with_cross_validation(self, all_data, x_function, y_function):
        list(self._repeated_k_fold_training(all_data, x_function, y_function))
        return self.model
    
    def text_preprocessing(self, x_train):
        from gensim.corpora import Dictionary
        tokenizer = Tokenizer(
            num_words=self.MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(x_train.values)

        x_train_values = [x.split(" ") for x in x_train.values]
        if self.use_lsi:
            self.dictionary = Dictionary(x_train_values)
            self.lsi = LsiModel(
                [self.dictionary.doc2bow(text) for text in x_train_values],
                id2word=self.dictionary,
                num_topics=self.EMBEDDING_DIM
            )
        if self.use_word2vec:
            self.word2vec = Word2Vec(x_train.values, size=self.EMBEDDING_DIM, min_count=5)

        return tokenizer
    
    def prepare_data(self, all_data):
        data = pd.DataFrame(data=all_data)

        x_data = self.x_function(data)
        self.tokenizer = self.text_preprocessing(x_data)
        x_data = self.tokenizer.texts_to_sequences(x_data.values)
        x_data = pad_sequences(x_data, maxlen=self.MAX_SEQUENCE_LENGTH)
        
        y_data = self.y_function(data)
        y_data = pd.get_dummies(y_data).values

        return x_data, y_data
    
    def train_model(self, x_train, y_train):
        if not self.model:
            self.logger.debug("Initiating model...")
            if self.use_word2vec:
                word2vec_embedding = self.word2vec.wv.get_keras_embedding()
            
            if self.use_lsi or self.use_word2vec:
                embedding_matrix_ns = np.random.random((len(self.tokenizer.word_index) + 1, self.EMBEDDING_DIM))
                for word, i in self.tokenizer.word_index.items():
                    try:
                        if self.word2vec:
                            embedding_vector = self.word2vec.wv.get_vector(word)
                        else:
                            vec_bow = self.dictionary.doc2bow(word.lower().split())
                            embedding_vector = self.lsi[vec_bow]
                            embedding_vector = [x[1] for x in embedding_vector]
                        
                        embedding_matrix_ns[i] = embedding_vector
                    except:
                        pass
                
                embedding = Embedding(
                    len(self.tokenizer.word_index) + 1, self.EMBEDDING_DIM,
                    input_length=x_train.shape[1], weights=[embedding_matrix_ns], trainable=False)
            else:
                embedding = Embedding(self.MAX_NB_WORDS, self.EMBEDDING_DIM, input_length=x_train.shape[1])
            
            input_layer = Input(shape=(x_train.shape[1],), dtype='int32')
            emb_layer = embedding(input_layer)

            output = getattr(nn_models, self.architecture)(
                embedding=emb_layer,
                memory_units=self.LSTM_LAYER_MEMORY_UNITS,
                dropout=self.dropout
            )
            
            self.model = Model(input_layer, Dense(self.OUTPUT_LAYER_VALUES, activation='softmax')(output))
            optimizer = Adam(1e-4)
            # Losses used: mean_squared_error, categorical_crossentropy, binary_crossentropy
            if self.keras_loss == 'dice_coef_loss':
                self.model.compile(loss=dice_coef_loss, optimizer=optimizer, metrics=[dice_coef])
            else:
                self.model.compile(
                    loss=self.keras_loss,
                    optimizer=optimizer,
                    metrics=[F1Score()] if self.use_f1_score else ['accuracy']
                )
            self.model.summary()
            plot_model(self.model, f'{self._get_model_filename()}.png')

        if len(self.test_data) and self.use_tests:
            validation_args = {
                'validation_data': self.prepare_data(self.test_data)
            }
        else:
            validation_args = {
                'validation_split': 0.1
            }

        class_weight = self._get_classes_weights(y_train)
        history = self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-4)],
            class_weight=class_weight,
            **validation_args
        )
        return self.model

    def _get_classes_weights(self, y_data):
        y_classes = np.unique(y_data, axis=0)
        classes_counts = list(map(lambda x: np.count_nonzero(y_data == x), y_classes))
        gcd = np.min(classes_counts)
        return dict([(i, v / gcd) for i, v in enumerate(classes_counts)])
    
    def test_model(self, model: Sequential, x_test, y_test):
        accr = model.evaluate(x_test, y_test)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    
    def save_model(self):
        self.model.save(self._get_model_filename())
    
    def classify_sentences(self, sentences):
        sentences = pd.DataFrame(data=sentences)[0]
        
        tokenizer = self.text_preprocessing(sentences)
        sentences = tokenizer.texts_to_sequences(sentences.values)
        sentences = pad_sequences(sentences, maxlen=self.MAX_SEQUENCE_LENGTH)
        
        y_labels = [0, 1]
        pred = self.model.predict(sentences)
        sentences_with_emotion = []
        for index, data in enumerate(pred):
            y_pred = y_labels[np.argmax(data)]
            if y_pred:
                sentences_with_emotion.append(index)
        
        return sentences_with_emotion, len(sentences)

    def plot_confusion_matrix(self, model, x_test, y_test):
        pred = model.predict(x_test)
        y_labels = np.unique(y_test)
        y_pred = []
        for data in pred:
            y_pred.append(y_labels[np.argmax(data)])

        self._plot_confusion_matrix(
            y_pred, y_test[:,1], y_labels,
            filename=f'{self.emotion_name}-{self.dataset_name}-lstm.png', title=f'{self.dataset_name} LSTM {self.emotion_name}')

    def _train_model(self, *args, **kwargs):
        x_data, y_data = self.prepare_data(self.train_data)
        self.text_clf = self.model
        return self.train_model(x_data, y_data)

    def _get_model_filename(self):
        return os.path.join("output", f'{self.emotion_name}-{self.dataset_name}-lstm.h5')


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    y_true = tf.cast(y_true, tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
