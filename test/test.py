import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.utils import np_utils, plot_model
from keras.datasets import mnist
from keras.datasets import fashion_mnist

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # convert class vector to binary class matries
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    x_train = x_train / 255
    x_test = x_test / 255
    #x_test = np.random_normal(x_test)
    return (x_train, y_train), (x_test, y_test)

def find_top_pred(scores):
    top_label_ix = np.argmax(scores)
    confidence = scores[0][top_label_ix]
    print('Predict Label: {}, Confidence: {}'.format(top_label_ix, confidence))

if __name__ == '__main__':
    # load mnist data
    (x_train, y_train), (x_test, y_test) = load_data()

    # load .h5 model
    model_name = 'cnn_demo_model'
    model = load_model('model_data/' + model_name + '.h5')

    output_op_name = model.output.op.name
    input_op_name = model.input.op.name
    
    # Compaer .h5 and .pb
    print("Anser Label: ", np.argmax(y_test[0]))
    print('======================== .h5 ========================')
    
    _input = np.expand_dims(x_test[0], 0)
    _input = _input.reshape(_input.shape[0], 28, 28, 1) # 此為Tensorflow寫法
    
    x = tf.placeholder(tf.float32, shape=model.get_input_shape_at(0))
    y = model(x)

    sess = K.get_session()
    orig_scores = sess.run(y, feed_dict={x: _input, K.learning_phase(): False})
    find_top_pred(orig_scores)
    K.clear_session()

    print('======================== .pb ========================')
    # ref: https://www.tensorflow.org/extend/tool_developers/#protocol_buffersp
    from tensorflow.core.framework import graph_pb2 
    from google.protobuf import text_format

    graph_def = graph_pb2.GraphDef() # This line creates an empty GraphDef object

    with open('model_data/forzen_cnn_demo_model.pb', 'rb') as f:
        # if is binary(.pb):
        graph_def.ParseFromString(f.read())

        # is is text(.pbtxt):
        #text_format.Merge(f.read(), output_graph_def)

        # tf.import_graph_def(): imports the graph from graph_def into the current default Graph.
        output= tf.import_graph_def(graph_def, name="")
    
    with tf.Session() as sess:
        x = sess.graph.get_tensor_by_name(input_op_name + ':0')
        y = sess.graph.get_tensor_by_name(output_op_name + ':0')

        new_scores = sess.run(y, feed_dict={x: _input, K.learning_phase(): False})
        find_top_pred(new_scores)
