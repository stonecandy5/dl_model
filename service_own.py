import os
import glob
import librosa
import time
import pyaudio
import sounddevice as sd
import numpy as np
import pandas as pd
import pickle
import scipy.io.wavfile as wavfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Create the directory for storing urban noise files
save_dir = "C:/Users/spong/myvenv/urban_noise"
os.makedirs(save_dir, exist_ok=True)

# Extract features from audio file
max_pad_len = 174

def extract_feature(file_path):
    print('파일 경로:', file_path)
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        print(mfccs.shape)

        # File path for saving the extracted features
        save_path = os.path.splitext(file_path)[0] + '.npy'

        # Save the extracted MFCC features as a npy file
        np.save(save_path, mfccs)
        print('MFCCs를 다음 경로에 저장했습니다:', save_path)

    except Exception as e:
        print("파일 분석 중 오류가 발생했습니다:", file_path)
        print(e)
        return None

# Microphone audio callback function
def audio_callback(indata, frames, callback_time, status):
    # Calculate current noise level
    rms = np.sqrt(np.mean(np.square(indata)))  # Calculate RMS value
    decibel = 20 * np.log10(rms + 1e-6)  # Calculate decibel value (+1e-6 to avoid log10(0))

    print(f"Current dB level: {decibel:.2f} dB")  # Print decibel value with 2 decimal places

    global recording_flag
    global recording_start_time
    global recording_data
    global current_noise_file

    if decibel >= -40:
        # Start recording
        if not recording_flag:
            recording_flag = True
            recording_start_time = callback_time.currentTime

        # Add current audio segment to recording data
        recording_data.append(indata.copy())

        # Check if recording duration exceeds 4 seconds
        if callback_time.currentTime - recording_start_time >= 4:
            # Stop recording and save the audio segment as WAV file
            recording_flag = False

            # Define filename based on timestamp
            current_time = int(time.time())
            filename = os.path.join(save_dir, f"noise_{current_time}.wav")

            # Convert the recording data to a numpy array
            recording_data = np.concatenate(recording_data, axis=0)

            # Save the recorded audio segment as WAV file if it exceeds 4 seconds
            if len(recording_data) >= 4 * samplerate:
                wavfile.write(filename, samplerate, recording_data)

                # Update the current_noise_file variable
                current_noise_file = filename

            # Clear the recording data buffer
            recording_data = []

    elif recording_flag:
        # Stop recording and save the audio segment as WAV file
        recording_flag = False

        # Define filename based on timestamp
        current_time = int(time.time())
        filename = os.path.join(save_dir, f"noise_{current_time}.wav")

        # Convert the recording data to a numpy array
        recording_data = np.concatenate(recording_data, axis=0)

        # Save the recorded audio segment as WAV file if it exceeds 4 seconds
        if len(recording_data) >= 4 * samplerate:
            wavfile.write(filename, samplerate, recording_data)

            # Update the current_noise_file variable
            current_noise_file = filename

        # Clear the recording data buffer
        recording_data = []

# Initialize recording variables
recording_flag = False
recording_start_time = 0.0
recording_data = []
current_noise_file = ""

# Open audio stream
samplerate = 44100  # Set sample rate
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate)

# Start audio stream
stream.start()

# Keep collecting audio while the program is running
try:
    while True:
        if current_noise_file:
            extract_feature(current_noise_file)
            # Reset the current_noise_file variable
            current_noise_file = ""
        time.sleep(0.1)
except KeyboardInterrupt:
    # Handle KeyboardInterrupt to stop the program gracefully
    pass

# Stop and close the audio stream
stream.stop()
stream.close()

# Load the trained model
model_path = 'C:\\Users\\spong\\myvenv\\dl_model\\sound_classifier_model'
model = keras.models.load_model(model_path)

# Prepare the input data
input_data = 'C:\\Users\\spong\\myvenv\\urban_noise\\noise_1684915383.npy'  # Specify the appropriate input data
input_data = np.load(input_data)  # Load the input data from the npy file

# Reshape the input data
input_data = np.expand_dims(input_data, axis=-1)  # Add the last dimension
input_data = np.expand_dims(input_data, axis=0)   # Add the batch dimension

# Model prediction
predictions = model.predict(input_data)
class_index = np.argmax(predictions, axis=1)
# Get the predicted label
predicted_label = str(class_index[0])

print("Predicted label:", predicted_label)
