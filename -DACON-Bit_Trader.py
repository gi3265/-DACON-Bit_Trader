#Import Library
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape, Flatten
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# Read Data
data_path = '../Big_Data/[DACON]Bit_Trader'
train_x_df = pd.read_csv(data_path  + "/train_x_df.csv")
train_y_df = pd.read_csv(data_path  + "/train_y_df.csv")
test_x_df = pd.read_csv(data_path  + "/test_x_df.csv")

# 시간 관계 상, train 데이터 상단의 300개 샘플를 구성하여 학습 및 추론
train_x_df = train_x_df[train_x_df.sample_id < 300]

#train_x_df와 train_y_df time열을 기준으로 concatenate하기(일단 생략하고 x_train만 가지고 학습 진행할 것. 1380 + 120 = 1500개 데이터로 학습하나 138만 가지고 학습하나 모델의 성능에 큰 차이를 미치지는 않을 것으로 추측하기 때문)


# 입력 받은 2차원 데이터 프레임을 3차원 numpy array로 변경하는 함수
def df2d_to_array3d(df_2d):
    feature_size = df_2d.iloc[:,2:].shape[1]
    time_size = len(df_2d.time.value_counts())
    sample_size = len(df_2d.sample_id.value_counts())
    array_3d = df_2d.iloc[:,2:].values.reshape([sample_size, time_size, feature_size])
    return array_3d


# 2차원 DF LSTM으로 학습하기 위해 3차원으로 변환시키기
train_x_array = df2d_to_array3d(train_x_df)   #(300, 1380, 10)
# valid_x_array = df2d_to_array3d(valid_x_df)
test_x_array = df2d_to_array3d(test_x_df)     #(529, 1380, 10)


#모델 구성(op1: many-to-one model, op2: many-to-many model(출력값 모양 바꾸고 return_sequences = True))
seq_len = 50
model = Sequential()
model.add(LSTM(64, activation='relu', return_sequences=True, input_shape = [seq_len, 1]))
model.add(Dense(240))
model.add(Dense(1))
model.add(Reshape([1, 1]))  #50개의 open값 입력되어 51번째 open값 '하나' 예측

model.compile(optimizer = 'adam', loss = 'mse')

model.summary()


# train_x_array데이터로 시계열 Windows 만들기
for idx in tqdm(range(train_x_array.shape[0])):
    seq_len = 50  # window_size와 같은 개념
    sequence_length = seq_len + 1

    windows = []
    for index in range(1380 - sequence_length):
        windows.append(train_x_array[idx, :, 1][index: index + sequence_length])

    # x_train, y_train 데이터 구성
    windows = np.array(windows) #1329 * 51의 2차원 배열
    x_train = windows[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = windows[:, -1]

    #Fit
    model.fit(x_train, y_train, validation_split=0.1, epochs = 3, batch_size = 1380) #batch_size = seq_len


# 손실값 시각화


# test_x_array데이터로 시계열 Windows 만들기
for idx in tqdm(range(test_x_array.shape[0])): #529번
    seq_len = 50
    sequence_length = seq_len + 1

    windows = []
    for index in range(1380 - sequence_length):
        windows.append(test_x_array[idx, :, 1][index: index + sequence_length])

    # x_test, y_test 데이터셋 구성
    windows = np.array(windows) #1329 * 51의 2차원 배열
    x_test = windows[:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = windows[:, -1]

    #Fit
    model.fit(x_test, y_test, epochs = 3, batch_size = 1380, verbose = 1)


# 모델 저장 및 로드
model.save('./my_model.h5')
model = tf.keras.models.load_model('./my_model.h5')


# Auto_Regressive한 Prediction 및 valid_pred_array에 예측 결과 기록
# 1. test_pred_array{예측값 모아두는 3차원 배열(120*1 2차원 배열 529개)} 만들기
test_pred_array = np.zeros([len(test_x_array), 120, 1])
# 2. test_x_array Windows 중 마지막 윈도우 추출해서 3차원 변환시켜 LSTM모델에 넣고 추론하기
window = windows[-1]
window_3d = np.reshape(window, (1, window.shape[0], 1))
pred = model.predict(window_3d)
# 3. 120분 중 처음 1분 예측값 test_pred_array에 기록
sample_id = 0  #나중에 for문으로 0~528까지 돌려야 할 변수
m = 0  #나중에 for문으로 0~119까지 돌려야 할 변수
test_pred_array[sample_id, m, :] = pred[sample_id, m, :]
# 4.



