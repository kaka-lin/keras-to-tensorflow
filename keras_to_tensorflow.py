import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam
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

def nn_model():
    model = Sequential()
    model.add(Dense(input_dim=28*28, units=689, activation='relu'))
    model.add(Dense(units=689, activation='relu'))
    model.add(Dense(units=689, activation='relu'))
    model.add(Dense(units=10, activation='softmax', name='output'))

    # loss='categorical_crossentropy'
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()

    return model

def train_model(model, x_train, y_train, batch_size, epochs):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    result = model.evaluate(x_train, y_train)
    print('\nTrain Acc: ', result[1])

    return model

def find_top_pred(scores):
    top_label_ix = np.argmax(scores)
    confidence = scores[0][top_label_ix]
    print('Predict Label: {}, Confidence: {}'.format(top_label_ix, confidence))

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """ Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.

    @param session:         The TensorFlow session to be frozen.
    @param keep_var_names:  A list of variable names that should not be frozen,
                            or None to freeze all the variables in the graph.
    @param output_names:    Names of the relevant graph outputs.
    @param clear_devices:   Remove the device directives from the graph for better portability.

    @return The frozen graph definition.

    """

    from tensorflow.python.framework.graph_util import convert_variables_to_constants

    graph = session.graph

    with graph.as_default():
        # show global variables
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))

        output_names = output_names or []
        #output_names += [v.op.name for v in tf.global_variables()]
        
        input_graph_def = graph.as_graph_def()

        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""

        #frozen_graph = convert_variables_to_constants(session, input_graph_def,
        #                                              output_names, freeze_var_names)

        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names)

        return frozen_graph

if __name__ == '__main__':
    # load mnist data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Creat NN model for mnist
    model = nn_model()
    model = train_model(model, x_train, y_train, 100, 10)

    result = model.evaluate(x_test, y_test)
    print('\nTest Acc: ', result[1])

    model.save('nn_model.h5')
    plot_model(model, to_file='nn_model.png', show_shapes=True)
    
    # load model and convert .h5 to .pb
    from keras import backend as K

    model = load_model('nn_model.h5')
    output_op_name = model.output.op.name
    input_op_name = model.input.op.name
    
    sess = K.get_session()
    frozen_graph = freeze_session(sess, output_names=[output_op_name])

    tf.train.write_graph(frozen_graph, "./", "my_model.pb", as_text=False)
    tf.train.write_graph(frozen_graph, "./", "my_model.pbtxt", as_text=True)

    # Compaer .h5 and .pb
    print("Anser Label: ", np.argmax(y_test[0]))
    print('======================== .h5 ========================')

    _input = np.expand_dims(x_test[0], 0) 

    x = tf.placeholder(tf.float32, shape=model.get_input_shape_at(0))
    y = model(x)

    orig_scores = sess.run(y, feed_dict={x: _input, K.learning_phase(): False})
    find_top_pred(orig_scores)

    K.clear_session()
    
    print('======================== .pb ========================')
    # ref: https://www.tensorflow.org/extend/tool_developers/#protocol_buffersp
    from tensorflow.core.framework import graph_pb2 
    from google.protobuf import text_format

    sess = tf.Session()
    K.set_session(sess)

    graph_def = graph_pb2.GraphDef() # This line creates an empty GraphDef object

    with open('my_model.pb', 'rb') as f:
        # if is binary(.pb):
        graph_def.ParseFromString(f.read())

        # is is text(.pbtxt):
        #text_format.Merge(f.read(), output_graph_def)

        # tf.import_graph_def(): imports the graph from graph_def into the current default Graph.
        output= tf.import_graph_def(graph_def, name="")
    
    
    x = sess.graph.get_tensor_by_name(input_op_name + ':0')
    y = sess.graph.get_tensor_by_name(output_op_name + ':0')

    new_scores = sess.run(y, feed_dict={x: _input, K.learning_phase(): False})
    find_top_pred(new_scores)

    K.clear_session()
