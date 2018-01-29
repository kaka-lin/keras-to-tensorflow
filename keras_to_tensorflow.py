import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model

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
    dir = os.path.abspath(os.path.dirname(__file__))
    
    model_path = dir + '/model_data'
    model_name = 'cnn_demo_model'
    model = load_model(model_path + '/' + model_name + '.h5')
    model.summary()
    
    output_op_name = model.output.op.name
    input_op_name = model.input.op.name

    sess = K.get_session()
    frozen_graph = freeze_session(sess, output_names=[output_op_name])

    tf.train.write_graph(frozen_graph, model_path, 'forzen_' + model_name + '.pb' , as_text=False)
    tf.train.write_graph(frozen_graph, model_path, 'forzen_' + model_name + '.pbtxt' , as_text=True)
