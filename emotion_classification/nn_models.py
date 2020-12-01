from keras.layers import (LSTM, Bidirectional, Conv1D, Dense, Dropout,
                          Embedding, Flatten, GlobalAveragePooling1D,
                          GlobalMaxPooling1D, Input, MaxPool1D,
                          concatenate)

def lstm_bi(embedding, memory_units, dropout):
    lstm_bi = Bidirectional(LSTM(
        memory_units,
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=dropout,
        # recurrent_regularizer=L1L2(1e-2, 1e-2),
        # kernel_regularizer=L1L2(1e-5, 1e-2),
        # bias_regularizer=L1L2(1e-2, 1e-2)
    ))(embedding)
    max_pooling = MaxPool1D()(lstm_bi)
    return Flatten()(max_pooling)

def cnn(embedding, memory_units, dropout):
    cnn_1 = Conv1D(memory_units * 2, 6)(embedding)
    cnn_1_dropout = Dropout(dropout)(cnn_1)
    max_pooling = MaxPool1D()(cnn_1_dropout)
    return Flatten()(max_pooling)

def lstm_cnn_bi(embedding, memory_units, dropout):
    lstm_bi = Bidirectional(LSTM(
        memory_units,
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=dropout,
    ))(embedding)
    cnn_1 = Conv1D(memory_units * 2, 6)(lstm_bi)
    return GlobalMaxPooling1D()(cnn_1)

def lstm_cnn_conc(embedding, memory_units, dropout):
    lstm_bi = Bidirectional(LSTM(
        memory_units,
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=dropout,
    ))(embedding)
    cnn_1 = Conv1D(memory_units * 2, 6)(embedding)
    cnn_1_dropout = Dropout(dropout)(cnn_1)
    conc_1 = concatenate([lstm_bi, cnn_1_dropout], axis=1)
    lstm_cnn_max_pooling_1 = MaxPool1D()(conc_1)
    return Flatten()(lstm_cnn_max_pooling_1)

def two_lstm_bi_cnn_conc(embedding, memory_units, dropout):
    lstm_cnn_conc_1 = lstm_cnn_bi(embedding, memory_units, dropout)
    lstm_cnn_conc_2 = lstm_cnn_bi(embedding, memory_units, dropout)
    return concatenate([lstm_cnn_conc_1, lstm_cnn_conc_2], axis=1)
