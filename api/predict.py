import random
import tensorflow as tf
import tensorflow_text as text


def predict(readme):

    reloaded_model = tf.saved_model.load('/home/sergi/Desktop/BSC/Biohackaton/SH/contentSH/pipeline/SH-BioinformaticsSoftware/api/bioinfoRepo_bert')
    reloaded_results = tf.sigmoid(reloaded_model(tf.constant([readme])))
    print(reloaded_results)
    print('Results from the saved model:')
    print(reloaded_results[0][0])
    score = float(reloaded_results[0][0])
    if score < 0.5:
        prediction = ["bioinformatics"]
    else:
        prediction = ["non-bioinformatics"]
    return prediction, score












