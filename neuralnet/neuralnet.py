#!/usr/bin/env python3

# Imports
import tensorflow as tf
import scipy as sc
from scipy.io import wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import python_speech_features as psf
import filetype
import os

tf.logging.set_verbosity(tf.logging.INFO)

def generate_features(filename, windowSize=40):
	"""
	Given audio filename as .mp3 or .wav, generates a set of MFCC in a 1d array
	Params:
		filename - Type {@filetype.guess(filename)}
		windowSize
	Return:
		myFeatures - Type list
	"""
	rate, data = wav.read(filename)
	# print(rate, len(data))
	myFeatures = np.transpose(psf.mfcc(data, rate, nfft=1200))
	plt.imshow(myFeatures, aspect=10, interpolation='none')
	plt.savefig('py.png')
	myFeatures = myFeatures.flatten
	# print(len(myFeatures))
	return myFeatures

def get_data(datafile="data"):
	# Hardcode in to data file
	for file in os.listdir(os.fsencode(datafile)):
		filename = os.fsdecode(file)
		if filename.endswith(".wav"): 
			filename = filename[:4]
		
	return x,y

def input_fn_train(): # returns x, y
	pass
def input_fn_eval(): # returns x, y
	pass

estimator = tf.estimator.Estimator()

estimator.train(input_fn=input_fn_train)
estimator.evaluate(input_fn=input_fn_eval)

# Predictions
estimator.predict(input_fn=input_fn_predict)