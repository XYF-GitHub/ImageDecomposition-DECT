# -*- coding: utf-8 -*-

import tensorflow as tf

def variable_summaries(var, var_name):
    with tf.name_scope(var_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def convo(input_layer, layer_name, filter_shape, stride, padStr):
    with tf.variable_scope(layer_name):
        input_shape = input_layer.get_shape().as_list()
        print(layer_name, " input shape:", input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        
        conv_w = tf.get_variable("weight", filter_shape, initializer = tf.truncated_normal_initializer(stddev = 0.1))
        conv_b = tf.get_variable("bias", filter_shape[3], initializer = tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_layer, conv_w, [1, stride, stride, 1], padStr)
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv_b))
        
        variable_summaries(conv_w, "conv_w")
        variable_summaries(conv_b, "conv_b")
        variable_summaries(relu, "relu")            
        return relu
            
def convo_noneRelu(input_layer, layer_name, filter_shape, stride, padStr):
    with tf.variable_scope(layer_name):
        input_shape = input_layer.get_shape().as_list()
        print(layer_name, " input shape:", input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        
        conv_w = tf.get_variable("weight", filter_shape, initializer = tf.truncated_normal_initializer(stddev = 0.1))
        conv_b = tf.get_variable("bias", filter_shape[3], initializer = tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_layer, conv_w, [1, stride, stride, 1], padStr)
        none_relu = tf.nn.bias_add(conv, conv_b)
        
        variable_summaries(conv_w, "conv_w")
        variable_summaries(conv_b, "conv_b")
        variable_summaries(none_relu, "none_relu")            
        return none_relu

def deconvo(input_layer, layer_name, filter_shape, out_img_size, stride, padStr):
    with tf.variable_scope(layer_name):
        input_shape = input_layer.get_shape().as_list()
        print(layer_name, " input shape:", input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        conv_w = tf.get_variable("weight", filter_shape, initializer = tf.truncated_normal_initializer(stddev = 0.1))
        input_shape = tf.shape(input_layer)
        output_shape = tf.stack([input_shape[0], out_img_size[0], out_img_size[1], filter_shape[2]])
        conv_b = tf.get_variable("bias", filter_shape[2], initializer = tf.constant_initializer(0.0))
        conv = tf.nn.conv2d_transpose(input_layer, conv_w, output_shape, [1, stride, stride, 1], padStr)
        deconv = tf.nn.bias_add(conv, conv_b)
        
        variable_summaries(conv_w, "conv_w")
        variable_summaries(conv_b, "conv_b")
        variable_summaries(deconv, "deconv")
        return deconv

def deconv_withRelu(input_layer, layer_name, filter_shape, out_img_size, stride, padStr):
    with tf.variable_scope(layer_name):
        input_shape = input_layer.get_shape().as_list()
        print(layer_name, " input shape:", input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        conv_w = tf.get_variable("weight", filter_shape, initializer = tf.truncated_normal_initializer(stddev = 0.1))
        input_shape = tf.shape(input_layer)
        output_shape = tf.stack([input_shape[0], out_img_size[0], out_img_size[1], filter_shape[2]])
        conv_b = tf.get_variable("bias", filter_shape[2], initializer = tf.constant_initializer(0.0))
        conv = tf.nn.conv2d_transpose(input_layer, conv_w, output_shape, [1, stride, stride, 1], padStr)
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv_b))
        
        variable_summaries(conv_w, "conv_w")
        variable_summaries(conv_b, "conv_b")
        variable_summaries(relu, "relu")
        return relu

def pooling(input_layer, layer_name, kernal_shape, stride, padStr):
    with tf.name_scope(layer_name):
        pool = tf.nn.max_pool(input_layer, kernal_shape, [1,stride,stride,1], padStr)
        return pool

def pooling_withmax(input_layer, layer_name, kernal_shape, stride, padStr):
    with tf.name_scope(layer_name):
        return tf.nn.max_pool_with_argmax(input_layer, kernal_shape, [1,stride,stride,1], padStr)
            
def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.stack(output_list)

def unpooling_layer2x2(x, layer_name, raveled_argmax, out_shape):
    with tf.name_scope(layer_name):
        argmax = unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
        output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

        height = tf.shape(output)[0]
        width = tf.shape(output)[1]
        channels = tf.shape(output)[2]

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

        t2 = tf.squeeze(argmax)
        t2 = tf.stack((t2[0], t2[1]), axis=0)
        t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

        t = tf.concat([t2, t1], 3)
        indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

        x1 = tf.squeeze(x)
        x1 = tf.reshape(x1, [-1, channels])
        x1 = tf.transpose(x1, perm=[1, 0])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)
            
def unpooling_layer2x2_batch(bottom, layer_name, argmax):
    with tf.name_scope(layer_name):
        bottom_shape = tf.shape(bottom)
        top_shape = [bottom_shape[0], bottom_shape[1] * 2, bottom_shape[2] * 2, bottom_shape[3]]

        batch_size = top_shape[0]
        height = top_shape[1]
        width = top_shape[2]
        channels = top_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        argmax = unravel_argmax(argmax, argmax_shape)

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [batch_size * (width // 2) * (height // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, batch_size, height // 2, width // 2, 1])
        t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

        t2 = tf.to_int64(tf.range(batch_size))
        t2 = tf.tile(t2, [channels * (width // 2) * (height // 2)])
        t2 = tf.reshape(t2, [-1, batch_size])
        t2 = tf.transpose(t2, perm=[1, 0])
        t2 = tf.reshape(t2, [batch_size, channels, height // 2, width // 2, 1])

        t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

        t = tf.concat([t2, t3, t1], 4)
        indices = tf.reshape(t, [(height // 2) * (width // 2) * channels * batch_size, 4])

        x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])
        return tf.scatter_nd(indices, values, tf.to_int64(top_shape))

def fullyconnect(input_layer, layer_name, input_size, output_size, regularizer):
    with tf.variable_scope(layer_name):
        fc_w = tf.get_variable("weight", [input_size, output_size], 
                               initializer = tf.truncated_normal_initializer(stddev = 0.1))
        if regularizer != 0:
            tf.add_to_collection('losses', regularizer(fc_w))                              
        fc_b = tf.get_variable("bias", [output_size], initializer = tf.constant_initializer(0.1))           
        fc = tf.nn.sigmoid(tf.matmul(input_layer, fc_w) + fc_b)
        #if tain: fc = tf.nn.dropout(fc, 0.5)
        return fc

def linearlyconnect(input_layer, layer_name, input_size, output_size, regularizer):
    with tf.variable_scope(layer_name):
        fc_w = tf.get_variable("weight", [input_size, output_size], 
                               initializer = tf.truncated_normal_initializer(stddev = 0.1))
        if regularizer != 0:
            tf.add_to_collection('losses', regularizer(fc_w))
            
        fc_b = tf.get_variable("bias", [output_size], initializer = tf.constant_initializer(0.1))
        logit = tf.matmul(input_layer, fc_w) + fc_b
        return logit