import random
import tensorflow as tf
import tensorflow_text as text

# def print_my_examples(inputs, results):
#   result_for_printing = [f'score: {results[i][0]:.6f}'
#                          for i in range(len(inputs))]
#   print(*result_for_printing, sep='\n')
#   print()

def predict(readme):
    # 🚧 generate randomly by now 🚧
    # prediction = random.choice(["bioinformatics", "non-bioinformatics"])
    # confidence = random.random()
    #  🚧  #

    reloaded_model = tf.saved_model.load('./bioinfoRepo_bert')

    reloaded_results = tf.sigmoid(reloaded_model(tf.constant(readme)))
    print(reloaded_results)
    print('Results from the saved model:')
    print(reloaded_results[0])
    if reloaded_results[0] < 0.5:
        prediction = ["bioinformatics"]
    else:
        prediction = ["non-bioinformatics"]

    return prediction, reloaded_results[0]