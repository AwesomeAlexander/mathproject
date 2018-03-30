import os
import scipy as sc
from scipy.io import wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import python_speech_features as psf
import filetype


#Given audio filename as .mp3 or .wav, generates a set of MFCC in a 1d array
def func(filename, windowSize=40):
    rate, data = wav.read(filename)
    # print(rate, len(data))
    myFeatures = np.transpose(psf.mfcc(data, rate, nfft=1200))
    plt.imshow(myFeatures, aspect=10, interpolation='none')
    plt.savefig('py.png')
    # myFeatures = myFeatures.flatten
    # print(len(myFeatures))
    return myFeatures

while True:
	filename = input("What file?\n")
	func(filename)