#Author: Andreas Neokleous
import os
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
from glob import glob
import re
import pandas as pd
import gc
from scipy.io import wavfile

from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.models import load_model
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import time
with tf.device("/gpu:0"):

	# The original sample rate of the dataset (*.wav files)
	ORIGINAL_SAMPLE_RATE = 16000

	# Sample rate used in training - lower sample rate means faster training
	NEW_SAMPLE_RATE = 8000

	class_labels =['eight', 'five', 'four', 'nine', 'one', 'seven', 'six', 'three', 'two', 'zero']

	#Test - Train Data Directories
	dirname = os.path.dirname(os.path.realpath(__file__))
	test_data_path = os.path.join(dirname, 'test')
	train_data_path = os.path.join(dirname, 'train')

	print(test_data_path)
	print(train_data_path)
	# GPU info
	def get_available_gpus():
	    local_device_protos = device_lib.list_local_devices()
	    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']
        # Search the training data location to find *.wav files and their label.
        # Returns labels list and files list
	def get_train_data(dirpath=train_data_path, extension='wav'):
		print('Loading training data from: ', dirpath)
		train_paths = glob(os.path.join(dirpath, r'*/*' + extension))
		labels = []

		file_names = []
		for path in train_paths:
			file_names.append(os.path.basename(path))
			labels.append(os.path.basename(os.path.dirname(path)))
		return labels, file_names

        # If the sample rate of a wav file is less than 16000, this function pads it
        # with 0s to make its length 16000.
	def wav_padding(samples):
		if len(samples) >= ORIGINAL_SAMPLE_RATE: return samples
		else: return np.pad(samples, pad_width=(ORIGINAL_SAMPLE_RATE - len(samples), 0), mode='constant', constant_values=(0, 0))

	#Uses to get metrics
	class GetMetrics(keras.callbacks.Callback):
		def on_train_begin(self, logs={}):
			self.acc = []
			self.val_acc = []

		def on_epoch_end(self, batch, logs={}):
			self.acc.append(logs.get('acc'))
			self.val_acc.append(logs.get('val_acc'))

	metrics = GetMetrics()

        # The following function calculates the logarithmic values of a spectrogram.
        # A spectrogram is an image that represents the spectrum of frequencies of
        # the *.wav files used in training and testing.
	def spectrogram_calculation(audio, sample_rate, window_size=20,step_size=10, eps=1e-10):
		nperseg = int(round(window_size * sample_rate / 1e3))
		noverlap = int(round(step_size * sample_rate / 1e3))
		freqs, times, spec = signal.spectrogram(audio,
		                                fs=sample_rate,
		                                window='hann',
		                                nperseg=nperseg,
		                                noverlap=noverlap,
		                                detrend=False)
		return freqs, times, np.log(spec.T.astype(np.float32) + eps)

        # Converts categorical variables (labels) into indicator variables.
	def label_convertion(labels):
		nlabels = []
		for label in labels:
			nlabels.append(label)
		return pd.get_dummies(pd.Series(nlabels))

	# This function prepares the test data to be fed into CNN
	# Returns a list with the file paths/names and an array with their spectograms
	def test_data_processing(dirpath=test_data_path, extension='wav'):
		test_paths = glob(os.path.join(dirpath, r'*/*' + extension))
		target = []
		spectrograms = []
		file_names = []
		for path in test_paths:
			sample_rate, samples = wavfile.read(path)
			samples = wav_padding(samples)
			resampled = signal.resample(samples, int(NEW_SAMPLE_RATE / sample_rate * samples.shape[0]))
			_, _, spectrogram = spectrogram_calculation(resampled, sample_rate=NEW_SAMPLE_RATE)
			spectrograms.append(spectrogram)
			file_names.append(path)
			target.append(os.path.basename(os.path.dirname(path)))
			#os.path.dirname(os.path.dirname(file))
		spectrograms = np.array(spectrograms)
		spectrograms = spectrograms.reshape(tuple(list(spectrograms.shape) + [1]))
		yield file_names, spectrograms, target


	if os.path.exists('./trained_model.h5'):
		print ('***Model exists***')
		print ('Using trained model. To retrain please delete the file ./trained_model.h5')
		model = load_model(os.path.join('.', 'trained_model.h5'))

		print(get_available_gpus())
		# Model Summary
		model.summary()
	if not os.path.exists('./trained_model.h5'):

		# Getting the labels and files to be used in the training
		labels, file_names = get_train_data()

		#Y for labels of the dataset
		Y = []
		#X for training dataset = spectograms
		X = []

		# zip() retruns an interator. Each element of the iterator is formed by 'lables' and 'file_names'.
		for label, file_name in zip(labels, file_names):
			#returns sample_rate and samples(wav file data) for each *.wav file
			sample_rate, samples = wavfile.read(os.path.join(train_data_path, label, file_name))
			#Pads waf file data of length > 16000 with 0s
			samples = wav_padding(samples)
			#Iterates the wav file data and resamples it to 8000
			#Saves label to Y and Calculates spectogram to save it to X
			for data in [samples]:
				resampled = signal.resample(data, int(NEW_SAMPLE_RATE / sample_rate * data.shape[0]))
				_, _, spectrogram = spectrogram_calculation(resampled, sample_rate=NEW_SAMPLE_RATE)
				Y.append(label)
				X.append(spectrogram)

		X = np.array(X)
		# Shape of X[x] before reshape (99,81)
		X = X.reshape(tuple(list(X.shape) + [1]))
		# Shape of X[x] after reshape (99,81,1). Spectogram resolution 99x81, black/white

		Y = label_convertion(Y)
		Y = Y.values
		Y = np.array(Y)

		# Splitting training dataset
		x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.18, random_state=2018)

		# Total samples: 22843
		# Training data: 15910 (~70% of total)
		# Validation data: 3493 (~15% of total)
		# Testing data in ./test folder: 3440 (~15% of total)

		print ('***No trained model exists***')
		print ('Training new model.')
		### Creating the CNN ###
		input_shape = (99, 81, 1)
		# 10 sounds. Spoken digits from 0 - 9
		nclass = 10
		input_layer = Input(shape=input_shape)

		model = Convolution2D(16, kernel_size=3, activation=activations.relu)(input_layer)

		model = MaxPooling2D(pool_size=(2, 2))(model)

		model = Convolution2D(32, kernel_size=3, activation=activations.relu)(model)

		model = MaxPooling2D(pool_size=(2, 2))(model)

		model = Dropout(rate=0.2)(model)
		model = Convolution2D(64, kernel_size=3, activation=activations.relu)(model)

		model = Flatten()(model)
		dense_layer = Dense(nclass, activation=activations.softmax)(model)
		model = models.Model(inputs=input_layer, outputs=dense_layer)

		# Compile model
		epochs = 10
		opt = optimizers.Adam() # Default: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,
		model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['accuracy'])

		print(get_available_gpus())
		# Model Summary
		model.summary()
		# Start timer
		start_time = time.time()
		# Fit model
		model.fit(x_train, y_train, batch_size=16, validation_data=(x_valid, y_valid), epochs=epochs, shuffle=True, verbose=2, callbacks=[metrics])
		# Get time
		total_train_time = time.time() - start_time
		print('Total training time in seconds: ')
		print(total_train_time)
		# Save model to file
		model.save(os.path.join('.', 'trained_model.h5'))

		#    Plotting Epoch/accuracy
		#    print (metrics.acc)
		#    plt.plot(metrics.acc)
		#    plt.plot(metrics.val_acc,color='red')
		#    plt.xlabel('epochs')
		#    plt.ylabel('accuracy')
		#    plt.show()

		#Final evaluation of the model
		score_val = model.evaluate(x_valid, y_valid, verbose=0)
		print ("Accuracy on Validation data:  %.2f%%" % (score_val[1]*100))
		score_train = model.evaluate(x_train, y_train, verbose=0)
		print ("Accuracy on Train data:  %.2f%%" % (score_train[1]*100))


	test_files = []
	predictions = []
	correct = []
	print ('Predicting test data ...')

	for file_names, spectrograms, target in test_data_processing():
		predicts = model.predict(spectrograms)
		predicts = np.argmax(predicts, axis=1)
		predicts = [class_labels[p] for p in predicts]
		test_files.extend(file_names)
		predictions.extend(predicts)
		correct.extend(target)

	number_of_corrects = 0;
	for x in range(len(predictions)):
		if (np.array_equal(correct[x],predicts[x])):
			number_of_corrects += 1
	score_test = number_of_corrects/len(predictions)
	print ("Accuracy on Test data:  %.2f%%" % (score_test*100))

	print ('Saving predictions to csv ...')
	df = pd.DataFrame(columns=['file_name', 'predicted_label'])
	df['file_name'] = test_files
	df['predicted_label'] = predictions
	df.to_csv(os.path.join('.', 'predicted_test_results.csv'), index=False)
