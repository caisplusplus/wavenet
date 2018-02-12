import numpy as np
import tensorflow as tf
from tqdm import tqdm
from wavenet.layers import _causal_linear, _output_linear, conv1d, dilated_conv1d
from train import create_network
import wavenet.utils
import os

from scipy.io import wavfile

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
# Can be increased to process multiple audio samples at once
batch_size = 1


bins = np.linspace(-1, 1, num_classes)

# Load in the overall network
sess, cost, trian_step, _, targets = create_network()

inputs = tf.placeholder(tf.float32, [batch_size, 1],
        name='inputs')

saver = tf.train.Saver()
restore_model = 'final_1007'
saver.restore(sess, 'weights/model_' + restore_model + '.ckpt')

##############################################################################
# We will make a modification to the network so that generating new samples is
# faster. This is just removes redundant convolution operators using dynamic
# programming. For where this algorithm came from and how it works please see
# https://github.com/tomlepaine/fast-wavenet
##############################################################################

count = 0
h = inputs

init_ops = []
push_ops = []
for b in range(num_blocks):
    for i in range(num_layers):
        rate = 2**i
        name = 'b{}-l{}'.format(b, i)
        if count == 0:
            state_size = 1
        else:
            state_size = num_hidden

        q = tf.FIFOQueue(rate,
                         dtypes=tf.float32,
                         shapes=(batch_size, state_size))
        init = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))

        state_ = q.dequeue()
        push = q.enqueue([h])
        init_ops.append(init)
        push_ops.append(push)

        h = _causal_linear(h, state_, name=name, activation=tf.nn.relu)
        count += 1

outputs = _output_linear(h)

out_ops = [tf.nn.softmax(outputs)]
out_ops.extend(push_ops)

sess.run(init_ops)

#################################
# LOAD DATA
#################################
num_time_samples = 40000
input_data, _ = wavenet.utils.make_batch(
        '/hdd/datasets/VCTK-Corpus/wav48/p225/p225_001.wav',
        num_time_samples)

# Get the first sample of the audio. Let the network predict the rest.
input_data = input_data[:, :1, :]
input_data = input_data.reshape(1, 1)

# Number of successive samples to generate
num_samples_to_gen = 9000

predictions = []
for step in range(num_samples_to_gen):
    if step % 100 == 0:
        print('Generating %i / %i' % (step, num_samples_to_gen))
    feed_dict = {inputs: input_data}
    output = sess.run(out_ops, feed_dict=feed_dict)[0]
    value = np.argmax(output[0, :])

    input_data = np.array(bins[value])[None, None]
    predictions.append(input_data)

# Join all of the predictions
final_pred = np.concatenate(predictions, axis=1)
final_pred = final_pred.flatten()

if not os.path.exists('results/'):
    os.makedirs('results/')

# Write the final saved data
wavfile.write('results/out.wav', 44100, final_pred)
