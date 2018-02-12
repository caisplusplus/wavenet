import numpy as np
import tensorflow as tf
from tqdm import tqdm
from wavenet.layers import _causal_linear, _output_linear, conv1d, dilated_conv1d
import wavenet.utils
import os


# VCTK audio just has one channel
num_channels = 1
# Audio output is a byte [0 - 255] that's 256 unique values
num_classes = 256
# An entire set of dilated convolution layers is called a block
# We can stack these blocks on top of each other to get better
# model capacity and receptive field size
num_blocks = 2
# The number of successive dilated convolutions we have per block
num_layers = 14
# The number of nodes per convolution
num_hidden = 128

# This is the number of audio points we will use per sample so we can have a
# set tensor shape for data input.
num_time_samples = 40000

#################################
# START BY DEFINING OUR NETWORK
#################################

def create_network():
    # Placeholders for X and Y data
    inputs = tf.placeholder(tf.float32, shape=(None, num_time_samples, num_channels))
    targets = tf.placeholder(tf.int32, shape=(None, num_time_samples))

    # Dialated convolutions will be recursively applied.
    h = inputs
    hs = []
    for b in range(num_blocks):
        for i in range(num_layers):
            rate = 2**i
            name = 'b{}-l{}'.format(b, i)
            h = dilated_conv1d(h, num_hidden, rate=rate, name=name)
            hs.append(h)

    outputs = conv1d(h,
                     num_classes,
                     filter_width=1,
                     gain=1.0,
                     activation=None,
                     bias=True)

    costs = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=outputs, labels=targets)
    cost = tf.reduce_mean(costs)

    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    return sess, cost, train_step, inputs, targets

if __name__ == '__main__':
    # (to make sure we have a location we can save the model to)
    if not os.path.exists('weights/'):
        os.makedirs('weights/')
    #################################
    # LOAD OUR DATA
    #################################

    take_count = 300
    data_loc = '/hdd/datasets/VCTK-Corpus/wav48'

    # If you're working with a large dataset that has the form
    # data_loc/*/*.wav
    #all_inputs, all_targets = wavenet.utils.make_batch_all(data_loc, take_count, num_time_samples)

    # If you're working with a single audio sample
    all_inputs, all_targets = wavenet.utils.make_batch_single(
            '/hdd/datasets/VCTK-Corpus/wav48/p225/p225_001.wav',
            num_time_samples)

    #################################
    # TIME TO TRAIN THE NETWORK
    #################################

    sess, cost, train_step, inputs, targets = create_network()

    # Actually train the network now
    saver = tf.train.Saver()

    base_weight_path = 'weights/'

    if len(all_inputs) == 1:
        print('Single mode')
        max_no_save = 100
        save_interval = 1000
    else:
        print('Batch mode')
        max_no_save = 50
        save_interval = 200

    try:
        epoch_num = 1
        should_stop = False
        combined_data = list(zip(all_inputs, all_targets))

        while not should_stop:
            # Only display progress bar if there is more than one sample being
            # processed
            iterate_obj = tqdm(combined_data) if len(combined_data) > 1 else combined_data

            for input_data, target_data in iterate_obj:
                feed_dict = {inputs: input_data, targets: target_data}
                loss, _ = sess.run([cost, train_step], feed_dict=feed_dict)

            print('Epoch %i, Loss: %.4f' % (epoch_num, loss))
            if loss < 1e-4:
                should_stop = True

            if epoch_num % save_interval == 0:
                print('Saving model')
                saver.save(sess, base_weight_path + 'model_%i.ckpt' % epoch_num)

            epoch_num += 1

    except KeyboardInterrupt:
        # So we can save if sigint
        print('Interrupted')

    if epoch_num > max_no_save:
        print('Final save')
        saver.save(sess, base_weight_path + 'model_final_%i.ckpt' % epoch_num)

