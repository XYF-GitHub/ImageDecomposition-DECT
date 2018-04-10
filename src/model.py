import tensorflow as tf
import layer_xyf

#input size
INPUT_W = 65
INPUT_H = 65
INPUT_DEPTH = 1

# model definition
#layer_1 input size 65*65   [filter_height, filter_width, in_channels, out_channels]
FILTER_1 = [5, 5, 1, 64]
STRIDE_1 = 2
PAD_1 = "SAME"

#layer_2 input_size 33*33
FILTER_2 = [5, 5, 64, 128]
STRIDE_2 = 2
PAD_2 = "SAME"

#layer_3 input_size 17*17
FILTER_3 = [5, 5, 128, 256]
STRIDE_3 = 2
PAD_3 = "SAME"

#layer_4 input_size 9*9
FILTER_4 = [9, 9, 256, 256]
STRIDE_4 = 1
PAD_4 = "VALID"

#layer_5 input_size 1*1*2048
FILTER_5 = [1, 1, 512, 1]
STRIDE_5 = 1
PAD_5 = "VALID"

class DecompositionModel(object):
    def __init__(self, sess, epoch, batch_size = 64, model_name = 'model'):
        self.sess = sess
        self.epoch = epoch
        self.input_width = INPUT_W
        self.input_height = INPUT_H
        self.batch_size = batch_size
        self.model_name = model_name
        
    def feedforward(self, input_L_batch, input_H_batch, regularizer = 0):
        layer_L1 = layer_xyf.convo(input_L_batch, "conv_L1", FILTER_1, STRIDE_1, PAD_1)
        layer_L2 = layer_xyf.convo(layer_L1, "conv_L2", FILTER_2, STRIDE_2, PAD_2)
        layer_L3 = layer_xyf.convo(layer_L2, "conv_L3", FILTER_3, STRIDE_3, PAD_3)
        layer_L4 = layer_xyf.convo(layer_L3, "conv_L4", FILTER_4, STRIDE_4, PAD_4)
        
        layer_H1 = layer_xyf.convo(input_H_batch, "conv_H1", FILTER_1, STRIDE_1, PAD_1)
        layer_H2 = layer_xyf.convo(layer_H1, "conv_H2", FILTER_2, STRIDE_2, PAD_2)
        layer_H3 = layer_xyf.convo(layer_H2, "conv_H3", FILTER_3, STRIDE_3, PAD_3)
        layer_H4 = layer_xyf.convo(layer_H3, "conv_H4", FILTER_4, STRIDE_4, PAD_4)

        combine_LH = tf.concat([layer_L4, layer_H4], 3)
        
        layer_bone = layer_xyf.convo_noneRelu(combine_LH, "conv_bone", FILTER_5, STRIDE_5, PAD_5)
        layer_tissue = layer_xyf.convo_noneRelu(combine_LH, "conv_tissue", FILTER_5, STRIDE_5, PAD_5)
        return layer_bone, layer_tissue
        
    def feedforward_test(self, input_L_batch, input_H_batch, regularizer = 0):
        layer_L1 = layer_xyf.convo(input_L_batch, "conv_L1", FILTER_1, 1, PAD_1)
        layer_L2 = layer_xyf.convo(layer_L1, "conv_L2", FILTER_2, 1, PAD_2)
        layer_L3 = layer_xyf.convo(layer_L2, "conv_L3", FILTER_3, 1, PAD_3)
        layer_L4 = layer_xyf.convo(layer_L3, "conv_L4", FILTER_4, STRIDE_4, PAD_4)
        
        layer_H1 = layer_xyf.convo(input_H_batch, "conv_H1", FILTER_1, 1, PAD_1)
        layer_H2 = layer_xyf.convo(layer_H1, "conv_H2", FILTER_2, 1, PAD_2)
        layer_H3 = layer_xyf.convo(layer_H2, "conv_H3", FILTER_3, 1, PAD_3)
        layer_H4 = layer_xyf.convo(layer_H3, "conv_H4", FILTER_4, STRIDE_4, PAD_4)

        combine_LH = tf.concat([layer_L4, layer_H4], 3)
        
        layer_bone = layer_xyf.convo_noneRelu(combine_LH, "conv_bone", FILTER_5, STRIDE_5, "SAME")
        layer_tissue = layer_xyf.convo_noneRelu(combine_LH, "conv_tissue", FILTER_5, STRIDE_5, "SAME")
        return layer_bone, layer_tissue

    def input_layer_width(self):
        return self.input_width

    def input_layer_height(self):
        return self.input_height
