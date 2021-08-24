import os
from pickle import load
import tensorflow as tf
from tensorflow import keras
import numpy as np
import librosa

#load saved neural network model
model = keras.models.load_model('neural_network.h5')

#load saved label encoder
labelencoder = load(open('encoder.pkl', 'rb'))

#input wav audio file path
filename="test2.wav"

#preprocess the audio file
audio, sample_rate = librosa.load(filename) 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
#Reshape MFCC feature to 2-D array
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)

# Predict the audio class type using the loaded model 
x_predict=model.predict(mfccs_scaled_features) 
predicted_label=np.argmax(x_predict,axis=1)

#use the label encoder to get the corresponding audio type label from the predicted class integer value
prediction_class = labelencoder.inverse_transform(predicted_label)[0] 
print(prediction_class)