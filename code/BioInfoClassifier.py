from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy
import pickle


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

    # loading
    with open('./tokenizer.pickle', 'rb') as handle:
        tok = pickle.load(handle)

    model = RNN()
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    model.load_weights('./data/my_weights')

    test_sequences = tok.texts_to_sequences(readMe)
    test_sequences_matrix = pad_sequences(test_sequences,maxlen=max_len)
    print(test_sequences_matrix)
    isBioinfo = model.predict(test_sequences_matrix)
    print(isBioinfo)


if __name__ == "__main__":
    readMe = ["# DAEi\nAutoencoders for Drug-Target Interaction Prediction\n\n- **Run AEi**:\n\n$ python AEi.py --path datasets/ --data_name Enzyme --epoches 300 --batch_size 256 --hidden_size 512 --reg 0.000001 --keep_rate 1.0 --lr 0.001 --min_loss 0.01 --cv 10 --loss_type square --mode dti\n\n- **Run DAEi**:\n\n$ python DAEi.py --path datasets/ --data_name Enzyme --epoches 300 --batch_size 256 --hidden_size 512 --regs [0.000001,0.000001,0.000001] --noise_level 0.00001 --lr 0.001 --min_loss 0.01 --cv 10 --loss_type square --mode dti\n\n\n## Parameter description：\n- path：Input data path.\n- data_name：Name of dataset: Enzyme, Ion Channel, GPCR, Nuclear Receptor\n- epoches：Number of epoches.\n- batch_size：Batch size.\n- hidden_size：Hidden layer size, also Embedding size.\n- reg: Regularization for L2.\n- keep_rate: Keep_rate of dropout.\n- lr: Learning rate.\n- min_loss: The minimum value for stopping loss function.\n- cv: K-fold Cross Validation.\n- mode: the mode for training: dti -> drug-target interactions; tdi -> target-drug interactions.\n"]
    transformReadMe(readMe)
