import os
import sys
import tensorflow as tf
import pprint

from model import DecompositionModel
from solver import nn_train, nn_feedforward

flags = tf.app.flags
flags.DEFINE_string("dataset", "data", ".tfRecord file [data]")
flags.DEFINE_string("mode", "train", "train, train_goon or feedforward [train]")
flags.DEFINE_string("model_name", "model", "The name of model [model]")
flags.DEFINE_integer("epoch", 30, "Epoch to train [500]")
flags.DEFINE_integer("batch_size", 10, "The size of batch images [10]")
flags.DEFINE_integer("model_step", 1000, "The number of iteration to save model [1000]")
flags.DEFINE_float("lr", 0.01, "The base learning rate [0.01]")
flags.DEFINE_string("checkpoint", "checkpoint", "The path of checkpoint")

FLAGS = flags.FLAGS

def main(_):
    os.chdir(sys.path[0])
    print("Current cwd: ", os.getcwd())
    
    FLAGS = flags.FLAGS
    FLAGS.goon = False
    FLAGS.output_model_dir = "../checkpoint"
    FLAGS.output_mat_dir = "../result"
    
    if not os.path.exists(FLAGS.output_model_dir):
        os.makedirs(FLAGS.output_model_dir)
    if not os.path.exists(FLAGS.output_mat_dir):
        os.makedirs(FLAGS.output_mat_dir)
    
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__flags)
    
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    
    with tf.Session(config = run_config) as sess:
        print("Building Decomposition model ...")
        nn = DecompositionModel(sess, epoch = FLAGS.epoch, 
                       batch_size = FLAGS.batch_size, 
                       model_name = FLAGS.model_name)
        
    if FLAGS.mode == "train":
        print("training ...")
        nn_train(nn, sess, FLAGS)
    elif FLAGS.mode == "feedforward":
        print("caculating ...")
        nn_feedforward(nn, sess, FLAGS)
    elif FLAGS.mode == "train_goon":
        print("go on training ...")
        FLAGS.goon = True
        nn_train(sess, FLAGS)
        
    pp.pprint(FLAGS.__flags)
                
    sess.close()
    print("Sess closed.")
    exit()

if __name__ == '__main__':
    tf.app.run()