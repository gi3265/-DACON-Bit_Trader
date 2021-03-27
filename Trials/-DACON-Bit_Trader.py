#Import Library
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape
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
train_y_df = train_y_df[train_y_df.sample_id < 300]
#train_x_df와 train_y_df time열을 기준으로 concatenate하기(일단 생략하고 x_train만 가지고 학습 진행할 것. 1380 + 120 = 1500개 데이터로 학습하나 1380개만 가지고 학습하나 모델의 성능에 큰 차이를 미치지는 않을 것으로 추측하기 때문)


# 입력 받은 2차원 데이터 프레임을 3차원 numpy array로 변경하는 함수
def df2d_to_array3d(df_2d):
    feature_size = df_2d.iloc[:,2:].shape[1]
    time_size = len(df_2d.time.value_counts())
    sample_size = len(df_2d.sample_id.value_counts())
    array_3d = df_2d.iloc[:,2:].values.reshape([sample_size, time_size, feature_size])
    return array_3d


# 2차원 DF LSTM으로 학습하기 위해 3차원으로 변환시키기
train_x_array = df2d_to_array3d(train_x_df)   #(300, 1380, 10)
train_y_array = df2d_to_array3d(train_y_df)   #(300, 120, 10)
test_x_array = df2d_to_array3d(test_x_df)     #(529, 1380, 10)


#모델 구성(op1: many-to-one model, op2: many-to-many model(output 모양 바꾸기, return_sequences = True, Window이동 단위 1에서 output 크기로 변경))
seq_len = 120
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences= True, input_shape = [seq_len, 1]))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))  #120개의 open값 입력되어 121번째 open값 '하나' 예측

model.compile(optimizer = 'adam', loss = 'mse')

model.summary()


# test_x_array에 대한 Auto_Regressive한 Prediction 및 valid_pred_array에 예측 결과 기록

# 1) test_pred_array{예측값 모아두는 3차원 배열(120*1 2차원 배열 529개)} 만들기
test_pred_array = np.zeros([len(test_x_array), 120, 1])

# 2) test_x_array로 시계열 Windows 만들기 -> 데이터셋 구성 -> 모델 학습 ||| 예측 -> test_pred_array에 기록 -> window_3d의 첫번째 값 삭제 -> test_pred_array와 window_3d 병합 -> model.predict()에 넣어 예측 -> ***

# test_x_array로 시계열 Windows 만들기
ep = 10
bs = 120
for idx in tqdm(range(test_x_array.shape[0])):  # 529번
    seq_len = 120
    sequence_length = seq_len + 1

    windows = []
    for index in range(1380 - sequence_length):
        windows.append(test_x_array[idx, :, 1][index: index + sequence_length])

    # x_test, y_test 데이터셋 구성
    windows = np.array(windows)  # 1329 * 121의 2차원 배열
    x_test = windows[:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = windows[:, -1]

    # Fit
    model.fit(x_test, y_test, epochs= ep, batch_size= bs, verbose=0, shuffle = True)

    # test_x_array Windows 중 마지막 윈도우 추출해서 3차원 변환시켜 LSTM모델에 넣고 Predict
    window = windows[-1, :-1]
    window_3d = np.reshape(window, (1, window.shape[0], 1))
    for m in range(window.shape[0]):
        # model.predict()에 window_3d 넣어 예측
        pred = model.predict(window_3d)

        # 120분 중 처음 1분 예측값 test_pred_array에 기록
        test_pred_array[idx, m, :] = pred

        # window_3d의 첫번째 분 값은 삭제한 window_3d_2nd 구성
        window_3d_2nd = window_3d[0, 1:, :]  # 119개

        # pred_target(prediction할 때마다 나오는 각각의 예측값들) 1차원 -> 2차원으로 구성
        pred_target = test_pred_array[idx, m, :]
        pred_target = np.reshape(pred_target, (pred_target.shape[0], 1))

        # test_pred_array와 window_3d_2nd 병합하여 모델에 입력할 새로운 window_3d 재구성
        window_3d = np.concatenate((window_3d_2nd, pred_target), axis=0)
        window_3d = window_3d.T
        window_3d = np.reshape(window_3d, (window_3d.shape[0], window_3d.shape[1], 1))


#test_pred_array에 채워진 (sample_id x번째 자료에 대한)120분 예측값 확인
x = 528
print(test_pred_array.shape)


# 모델 저장 및 로드
model.save('./my_model.h5')
model = tf.keras.models.load_model('./my_model.h5')


# 모델 평가: test_x데이터로 예측하는 방식을 입력값(train_x)에 대한 예측값과 실제값(train_y_array) 비교를 통해 평가
# x_train을 모델에 입력해서 나온 예측값 담을 train_pred_array 구성
train_pred_array = np.zeros([len(train_x_array), 120, 1])

# train_x_array데이터로 시계열 Windows 만들기
for idx in tqdm(range(train_x_array.shape[0])):
    seq_len = 120  # window_size와 같은 개념
    sequence_length = seq_len + 1

    windows = []
    for index in range(1380 - sequence_length):
        windows.append(train_x_array[idx, :, 1][index: index + sequence_length])

    # x_train, y_train 데이터 구성
    windows = np.array(windows) #1329 * 121의 2차원 배열
    x_train = windows[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = windows[:, -1]

    #Fit
    history = model.fit(x_train, y_train, validation_split=0.1, epochs = ep, batch_size = bs)

    # # test_x_array Windows 중 마지막 윈도우 추출해서 3차원 변환시켜 LSTM모델에 넣고 Predict
    # window = windows[-1, :-1]
    # window_3d = np.reshape(window, (1, window.shape[0], 1))
    # for m in range(window.shape[0]):
    #     # model.predict()에 window_3d 넣어 예측
    #     pred = model.predict(window_3d)
    #
    #     # 120분 중 처음 1분 예측값 test_pred_array에 기록
    #     test_pred_array[idx, m, :] = pred
    #
    #     # window_3d의 첫번째 분 값은 삭제한 window_3d_2nd 구성
    #     window_3d_2nd = window_3d[0, 1:, :]  # 119개
    #
    #     # pred_target(prediction할 때마다 나오는 각각의 예측값들) 1차원 -> 2차원으로 구성
    #     pred_target = test_pred_array[idx, m, :]
    #     pred_target = np.reshape(pred_target, (pred_target.shape[0], 1))
    #
    #     # test_pred_array와 window_3d_2nd 병합하여 모델에 입력할 새로운 window_3d 재구성
    #     window_3d = np.concatenate((window_3d_2nd, pred_target), axis=0)
    #     window_3d = window_3d.T
    #     window_3d = np.reshape(window_3d, (window_3d.shape[0], window_3d.shape[1], 1))


# train 샘플 훈련 성과 시각화해보기
# 1) 입력 series와 출력 series를 연속적으로 연결하여 시각적으로 보여주는 코드 정의
def plot_series(x_series, y_series):
    plt.plot(x_series, label = 'input_series')
    plt.plot(np.arange(len(x_series), len(x_series)+len(y_series)),
             y_series, label = 'output_series')
    plt.axhline(1, c = 'red')
    plt.legend()

# 2) train data 중 sample_id idx에 해당하는 x_series로 모델을 학습한 후 y_series를 추론
idx = 500
x_series = train_x_array[idx,:,1]
y_series = train_y_array[idx,:,1]
plt.plot(x_series, y_series)
plt.plot(np.arange(1380, 1380+120), preds, label = 'prediction')
plt.legend()
plt.show() # 한눈에 봐도 학습이 전혀 되지 않고 있다는 것 알 수 있음

# 손실값 시각화
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# 매수 시점, 매수 비율 표 만들기
# 1) train_pred_array 3차원에서 2차원으로 바꾸기
pred_array_2d = np.zeros([test_pred_array.shape[0], 120])

for idx in tqdm(range(test_pred_array.shape[0])):
    pred_array_2d[idx, :] = test_pred_array[idx, :, 0]

# 2) 예측값을 재해석하여 submission 표를 작성하는 함수 정의
def array_to_submission(pred_array):
    submission = pd.DataFrame(np.zeros([pred_array.shape[0], 2], np.int64),
                              columns=['buy_quantity', 'sell_time'])
    submission = submission.reset_index()
    buy_price = []
    for idx, sell_time in enumerate(np.argmax(pred_array, axis=1)):
        buy_price.append(pred_array[idx, sell_time])
    buy_price = np.array(buy_price)
    submission.loc[:, 'buy_quantity'] = (buy_price > 1.15) * 1
    submission['sell_time'] = np.argmax(pred_array, axis=1)
    submission.columns = ['sample_id', 'buy_quantity', 'sell_time']
    return submission

final_submission = array_to_submission(pred_array_2d)

# 전체 300가지 sample에 대해 _가지 case에서 115% 이상 상승한다고 추론함.
final_submission.buy_quantity.value_counts()


# final_submission csv파일로 저장
final_submission.to_csv('./submission.csv', index = False)



# many-to-many 모델로 다시 돌려보기(input: 120, output: 10 정도?), optimizer을 rmsprop으로 설정해보기,
# 시가에 임의의 상수 곱해서 증폭된 값으로 입력해보기
# NLP 119페이지 Earlystopping 적용
# 모델의 마지막 레이어였던 reshape는 필요 없는 것 같아 삭제했음
# model.add(TimeDistributed(Dense(1)))??


# epoch 수 늘리고 batch_size 줄여서 다시 학습
# LSTM 레이어 조절해보기(2번째 레이어 Bidirectional로?)
# shuffle= True 효과 있을까?

