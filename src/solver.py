import os
import scipy.io
import time
import tensorflow as tf
import numpy as np
import math

LEARNING_RATE_DECAY = 0.85
MOVING_AVERAGE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001

NUM_SAMPLE = 2454300

SUMMARY_DIR = "../log"
SUMMARY_STEP = 100

INPUT_H = 65
INPUT_W = 65
INPUT_CHANNEL = 1

OUTPUT_H = 1
OUTPUT_W = 1
OUTPUT_CHANNEL = 1

def nn_train(nn, sess, config):
    # read data
    print ("Train, loading .tfrecords data...", config.dataset)
    if not os.path.exists(config.dataset):
        raise Exception("training data not find.")
        return
        
    config.sampleNum = NUM_SAMPLE
    config.iteration = math.ceil( config.sampleNum / config.batch_size * config.epoch )
        
    time_str = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
    
    train_file_path = os.path.join(config.dataset, "train_sample_batches_*")
    train_files = tf.train.match_filenames_once(train_file_path)
    filename_queue = tf.train.string_input_producer(train_files, shuffle = True)
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, 
                                       features = {'train_x_low': tf.FixedLenFeature([], tf.string),
                                                   'train_x_high': tf.FixedLenFeature([], tf.string),
                                                   'train_y_bone': tf.FixedLenFeature([], tf.string),
                                                   'train_y_tissue': tf.FixedLenFeature([], tf.string)})
    
    train_x_low = tf.decode_raw(features['train_x_low'], tf.float32)
    train_x_high = tf.decode_raw(features['train_x_high'], tf.float32)
    train_y_bone = tf.decode_raw(features['train_y_bone'], tf.float32)
    train_y_tissue = tf.decode_raw(features['train_y_tissue'], tf.float32)
    
    train_x_low = tf.reshape(train_x_low, [INPUT_H, INPUT_W, INPUT_CHANNEL])
    train_x_high = tf.reshape(train_x_high, [INPUT_H, INPUT_W, INPUT_CHANNEL])
    train_y_bone = tf.reshape(train_y_bone, [OUTPUT_H, OUTPUT_W, OUTPUT_CHANNEL])
    train_y_tissue = tf.reshape(train_y_tissue, [OUTPUT_H, OUTPUT_W, OUTPUT_CHANNEL])
    
    train_x_low_batch, train_x_high_batch, train_y_bone_batch, train_y_tissue_batch = tf.train.shuffle_batch([train_x_low, train_x_high, train_y_bone, train_y_tissue], 
                                                    batch_size = config.batch_size, 
                                                    capacity = config.batch_size*3 + 1000,
                                                    min_after_dequeue = 1000)
    # loss function define
    ouput_bone_batch, ouput_tissue_batch = nn.feedforward(train_x_low_batch, train_x_high_batch, 0)
    global_step = tf.Variable(0, trainable = False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    mse_loss = tf.reduce_mean(tf.square(ouput_bone_batch - train_y_bone_batch) + tf.square(ouput_tissue_batch - train_y_tissue_batch))
    tf.add_to_collection('losses', mse_loss)
    tf.summary.scalar('MSE_losses', mse_loss)
    
    learning_rate = tf.train.exponential_decay(config.lr, global_step, 
                                               config.sampleNum / config.batch_size, LEARNING_RATE_DECAY)    
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse_loss, global_step = global_step)
                    
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name = 'train')
    
    merged = tf.summary.merge_all()
    summary_path = os.path.join(SUMMARY_DIR, time_str)
    os.mkdir(summary_path)
    summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
    
    saver = tf.train.Saver()
    if config.goon:
        if not os.path.exists(config.checkpoint):
            raise Exception("checkpoint path not find.")
            return
        print("Loading trained model... ", config.checkpoint)
        ckpt = tf.train.get_checkpoint_state(config.checkpoint)
        saver.retore(sess, ckpt)
    else:
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        #with tf.Session() as sess:
        #init = tf.global_variables_initializer()
        
    save_model_path = os.path.join(config.output_model_dir, time_str)
    os.mkdir(save_model_path)
	
    print("start training sess...")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    start_time = time.time()
    for i in range(config.iteration):
        batch_start_time = time.time()
        summary, _, loss_value, step, learningRate = sess.run([merged, train_op, mse_loss, global_step, learning_rate])
        batch_end_time = time.time()
        sec_per_batch = batch_end_time - batch_start_time
        if step % SUMMARY_STEP == 0:
            summary_writer.add_summary(summary, step//SUMMARY_STEP)
        if step % config.model_step == 0:
            print("Saving model (after %d iteration)... " %(step))
            saver.save(sess, os.path.join(save_model_path, config.model_name + ".ckpt"), global_step = global_step)
        print("sec/batch(%d) %gs, global step %d batches, training epoch %d/%d, learningRate %g, loss on training is %g" 
              % (config.batch_size, sec_per_batch, step, i*config.batch_size / config.sampleNum,
                 config.epoch, learningRate, loss_value))
        print("Elapsed time: %g" %(batch_end_time - start_time))
        
    coord.request_stop()
    coord.join(threads)
    summary_writer.close()
    print("Train done. ")
    print("Saving model... ", save_model_path)
    saver.save(sess, os.path.join(save_model_path, config.model_name + ".ckpt"), global_step = global_step)

def nn_feedforward(nn, sess, config):
    print ("Feed forward, loading .mat data...", config.dataset)
    if not os.path.exists(config.dataset):
        raise Exception(".mat file not find.")
        return
    
    mat = scipy.io.loadmat(config.dataset)
    input_xl = mat['I_L']
    input_xh = mat['I_H']

    dim_h = input_xl.shape[0]
    dim_w = input_xl.shape[1]
    if input_xl.ndim < 3:
        input_xl = input_xl.reshape(dim_h, dim_w, 1)
        input_xh = input_xh.reshape(dim_h, dim_w, 1)
        
    batch_size = input_xl.shape[2]
    print('image num: ', batch_size)
    print('input type: ', input_xl.dtype)

    x_L = tf.placeholder(tf.float32, [batch_size, dim_h, dim_w, 1], name = 'xl-input')
    x_H = tf.placeholder(tf.float32, [batch_size, dim_h, dim_w, 1], name = 'xh-input')
    y_bone, y_tissue = nn.feedforward_test(x_L, x_H, None)
    
    output_dim = 0
    I_bone = np.array([], dtype = np.float32)
    I_tissue = np.array([], dtype = np.float32)
	
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    
    ckpt = tf.train.get_checkpoint_state(config.checkpoint)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-'[-1])
        print('global step: ', global_step)
        for i in range(batch_size):
            xl = input_xl[:,:,i].reshape([1, dim_w, dim_h, 1])
            xh = input_xh[:,:,i].reshape([1, dim_w, dim_h, 1])
            print('xl: ', xl.shape)
            output_bone, output_tissue = sess.run([y_bone, y_tissue], feed_dict={x_L: xl, x_H: xh})
            print('TF Done')
            if i == 0:
                output_shape = output_bone.shape
                print("output shape: ", output_shape)
                output_dim = output_bone.shape[1]
                I_bone = np.zeros((output_dim, output_dim, batch_size), dtype = np.float32)
                I_tissue = np.zeros((output_dim, output_dim, batch_size), dtype = np.float32)
                
            output_bone.resize(output_dim, output_dim)
            output_tissue.resize(output_dim, output_dim)
            I_bone[:,:,i] = output_bone
            I_tissue[:,:,i] = output_tissue

        result_fileName = config.output_mat_dir + '/' + 'result_' \
                         + config.model_name + '.mat'
        scipy.io.savemat(result_fileName, {'I_bone':I_bone, 'I_tissue':I_tissue})
        print('Feed forward done. Result: ', result_fileName)
    else:
        print('No checkpoint file found.')
        return
