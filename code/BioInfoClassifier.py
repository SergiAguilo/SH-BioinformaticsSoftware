from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy


def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

def transformReadMe(readMe):
    global max_words
    global max_len
    max_words = 10000
    max_len = 1500
    tok = Tokenizer(num_words=max_words)

    model = RNN()
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    model.load_weights('../data/my_weights')

    test_sequences = tok.texts_to_sequences(readMe)
    test_sequences_matrix = pad_sequences(test_sequences,maxlen=max_len)
    isBioinfo = model.predict(readMe)
    print(isBioinfo)


if __name__ == "__main__":
    readMe = ['This is a tool for genomics assembly from the bioinformatics department of biology']
    transformReadMe(readMe)
