import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences
import pickle


def RNN():
    """
    Creates a Recurrent Neural Networks (RNN)

    Returns
    -------
    Model
        The resulting neural network.
    """
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

def run_model(readme_dict: dict):
    """
    Trains the neural network model on GitHub readmes
    to determine whether a repostiroy is related to bioinformatics.

    Parameters
    ----------
    readme_dict: dict
        The readme_dict object consisting of readmes as keys and
        boolean value where True == bioinformatics as values.
        
    Returns
    -------
    Tokenizer
        Tokenizer built on training dataset.
    Model
        Resulting trained Recurrent Neural Network model.
    list
        Test data containing readmes.
    list
        Test data containing boolean of bioinformatics topic.
    """
    is_bioinformatics = list(readme_dict.values())

    le = LabelEncoder()
    is_bioinformatics = le.fit_transform(is_bioinformatics)
    is_bioinformatics = is_bioinformatics.reshape(-1,1)

    # Model

    X_train,X_test,Y_train,Y_test = train_test_split(list(readme_dict.keys()),is_bioinformatics,test_size=0.15)

    global max_words
    global max_len
    max_words = 10000
    max_len = 1500
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = pad_sequences(sequences,maxlen=max_len)


    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)


    model = RNN()
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

    model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
            validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
    
    model.save_weights('../data/my_weights')

    return tok, model, X_test, Y_test


def evaluation(tok: Tokenizer, model: Model, readme_test: list, is_bioinformatics_test: list):
    """
    Prints the loss and accuracy of the trained model using readmes and their associated bioinformatics status as test data.

    Parameters
    ----------
    tok: Tokenizer
        Tokenizer built on training dataset.
    model: Model
        Trained Recurrent Neural Network model.
    readme_test: list
        Test data containing readmes.
    is_bioinformatics_test: list
        Test data containing boolean of bioinformatics topic.
    """
    test_sequences = tok.texts_to_sequences(readme_test)
    test_sequences_matrix = pad_sequences(test_sequences,maxlen=max_len)

    accr = model.evaluate(test_sequences_matrix,is_bioinformatics_test)

    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

def load_data(bioinformatics_readme_path: str, non_bioinformatics_readme_path: str) -> dict:
    """
    Loads readme JSON files from bioinformatics repositories and non-bioinformatics repositories
    into one dict that keeps track whether the readme is related to bioinformatics.

    Parameters
    ----------
    bioinformatics_readme_path: str
        Input file path for bioinformatics readme JSON.
    non_bioinformatics_readme_path: str
        Input file path for non-bioinformatics readme JSON.

    Returns
    -------
    dict
        Dict containing all readmes as keys and a boolean value (True = bioinformatics).
    """
    ## Bioinformatics repos
    with open(bioinformatics_readme_path) as f:
        fileBio=json.load(f)

    ## Non-bioinformatics repos
    with open(non_bioinformatics_readme_path) as f:
        fileNonBio=json.load(f)

    readme_dict = {}

    for key in fileBio:
        readme_dict[fileBio[key]] = True
    for key in fileNonBio:
        readme_dict[fileNonBio[key]] = False

    return readme_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ib', '--input_bioinformatics', type=str,
        help="Path to the json dict containing repository data related to bioinformatics.")
    parser.add_argument('-in', '--input_not_bioinformatics', type=str,
        help="Path to the json dict containing repository data not related to bioinformatics.")
    args = parser.parse_args()

    readme_dict = load_data(args.input_bioinformatics, args.input_not_bioinformatics)
    tok, model, X_test, Y_test = run_model(readme_dict)
    evaluation(tok, model, X_test, Y_test)