import os
import time
import pyaudio
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile

# Create the directory for storing urban noise files
save_dir = "C:/Users/spong/myvenv/urban_noise"
os.makedirs(save_dir, exist_ok=True)

# Microphone audio callback function
def audio_callback(indata, frames, callback_time, status):
    # Calculate current noise level
    rms = np.sqrt(np.mean(np.square(indata)))  # Calculate RMS value
    decibel = 20 * np.log10(rms + 1e-6)  # Calculate decibel value (+1e-6 to avoid log10(0))

    print(f"Current dB level: {decibel:.2f} dB")  # Print decibel value with 2 decimal places

    global recording_flag
    global recording_start_time
    global recording_data

    if decibel >= -40:
        # Start recording
        if not recording_flag:
            recording_flag = True
            recording_start_time = callback_time.currentTime

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

        # Clear the recording data buffer
        recording_data = []

    if recording_flag:
        # Append the current audio data to the recording buffer
        recording_data.append(indata.copy())

# Initialize recording variables
recording_flag = False
recording_start_time = 0.0
recording_data = []

# Open audio stream
samplerate = 44100  # Set sample rate
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate)

# Start audio stream
stream.start()

# Keep collecting audio while the program is running
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    # Handle KeyboardInterrupt to stop the program gracefully
    pass

# Stop and close the audio stream
stream.stop()
stream.close()
