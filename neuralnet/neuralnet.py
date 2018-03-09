#!/usr/bin/env python3

# Imports
import tensorflow as tf
import math


# Get input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp/data", one_hot=True)

# Consts
NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def model(data,*layers):
	"""
	Defining the Neural Net Model, to be run at a later time.

	Args:
		data: Input data to first layer
		*layers: Layers, @Int size of layer.
				First is input (with appropriate data to match),
				last is output (should be `NUM_CLASSES`),
				and the rest in-between are hidden layers.

	Returns:
		data: Output tensor computed through neural net
	"""
	before = IMAGE_PIXELS

	# Loop through, making layers and initialize
	for i in range(len(layers)-1):
		with tf.name_scope('hidden'+(i+1)):
			weights = tf.Variable(
				initial_value = tf.truncated_normal(
					[ before, layers[i] ],
					stddev = 1.0 / math.sqrt(float(IMAGE_PIXELS)) # TODO: Figure this out
					),
				name = 'weights'
			)
			biases = tf.Variable(
				initial_value = tf.zeros([layers[i]]),
				name = 'biases'
			)
			data = tf.nn.relu(tf.matmul(data, weights) + biases)
		# Set this for next iteration to save data
		before = layers[i]
	
	# Output layer - special
	with tf.name_scope('softmax_linear'):
		weights = tf.Variable(
			initial_value = tf.truncated_normal(
				[ before, layers[i] ],
				stddev = 1.0 / math.sqrt(float(IMAGE_PIXELS)) # TODO: Figure this out
				),
			name = 'weights'
		)
		biases = tf.Variable(
			initial_value = tf.zeros([layers[i]]),
			name = 'biases'
		)
		data = tf.matmul(data, weights) + biases # No ReLU applied here

	# Return
	return data