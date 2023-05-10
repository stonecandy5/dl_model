import os
import glob
import librosa
import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

max_pad_len = 174

def extract_feature(file_name):
    print('file name :', file_name)
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0), (0, pad_width)), mode='constant')
        print(mfccs.shape)

        # 파일 이름에서 확장자 제거해 저장위치 생성
        save_path = os.path.splitext(file_name)[0] + '.npy'

        #추출한 mfcc 특징을 npy 배열로 저장
        np.save(save_path, mfccs)
        print('MFCCs saved to:', save_path)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        print(e)
        return None
    

    
#     return padded_mfccs
    return mfccs

#extract_feature('C:\\Users\\asgc\\deep\\sound\\101415-3-0-2.wav')

