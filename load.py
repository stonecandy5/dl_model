import tensorflow as tf
import torch

model_path = 'C:\\Users\\asgc\\deep\\sound\\sound_classifier_model'


model = tf.saved_model.load(model_path)

# 입력 데이터 준비
input_data = 'C:\\Users\\asgc\\deep\\sound\\101415-3-0-2.npy' # 적절한 입력 데이터를 준비하세요

# 모델 예측
predictions = model(input_data)

# 예측 결과 확인
print(predictions)
