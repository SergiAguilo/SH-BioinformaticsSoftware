import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences


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

def runModel(X, Y):

    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = Y.reshape(-1,1)

    # Model
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

    global max_words
    global max_len
    max_words = 10000
    max_len = 1500
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = pad_sequences(sequences,maxlen=max_len)

    model = RNN()
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

    model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
            validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
    
    model.save_weights('../data/my_weights')

    return tok, model, X_test, Y_test


def evaluation(tok, model, X_test, Y_test):
    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = pad_sequences(test_sequences,maxlen=max_len)

    accr = model.evaluate(test_sequences_matrix,Y_test)

    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

def loadData():
    # Load Data

    ## Bioinformatics repos
    with open('../data/bio-topic-with-readme.json') as f:
        fileBio=json.load(f)

    listBioReadMe = [i for i in fileBio.values()]
    listIsBioinfo = ['isBioinfo' for i in range(len(listBioReadMe))]

    ## Non-bioinformatics repos
    with open('../data/non-bio-topic-with-readme.json') as f:
        fileNonBio=json.load(f)

    listNonBioReadMe = [i for i in fileNonBio.values()]
    listIsNotBioinfo = ['isNotBioinfo' for i in range(len(listNonBioReadMe))]

    # Group all together
    listBioinfoTrueFalse = listIsNotBioinfo + listIsBioinfo
    listAllReadme = listNonBioReadMe + listBioReadMe
    return listBioinfoTrueFalse, listAllReadme


if __name__ == "__main__":
   listBioinfoTrueFalse, listAllReadme = loadData()
   tok, model, X_test, Y_test = runModel(listAllReadme, listBioinfoTrueFalse)
   evaluation(tok, model, X_test, Y_test)
