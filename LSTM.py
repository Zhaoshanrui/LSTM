# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 14:22:18 2024

@author: DELL
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

#创建配置类，将LSTM的各个超参数声明为变量，便于后续使用
class Config():
    timestep = 7  # 时间步长，滑动窗口大小
    feature_size = 1 # 每个步长对应的特征数量，这里只使用1维，每天的价格数据
    batch_size = 1 # 批次大小
    output_size = 1 # 单输出任务，输出层为1，预测未来1天的价格
    hidden_size = 128 # 隐藏层大小
    num_layers = 1 # lstm的层数
    learning_rate = 0.0001 # 学习率
    epochs = 500 # 迭代轮数
    model_name = 'lstm' # 模型名
    best_loss = 0  # 记录损失
    activation = 'relu' # 定义激活函数
config = Config()

#读取数据
data = pd.read_excel(r"E:/桌面/工作簿2.xlsx", index_col='日期')
del data['品类']
sell = data['销量'].values

#划分数据集
train, test = train_test_split(sell, test_size=0.1, shuffle=False) 
##shuffle 参数的作用是控制数据在分割前是否需要随机打乱
##当设置为 True 时，数据会被随机打乱，然后按照指定的比例分割成训练集和测试集
##当 shuffle=False 时，数据将按照它们原有的顺序直接分割。这意味着数据集中的元素将保持它们在原始数据集中的相对顺序
##在时间序列数据的情况下，这通常是必须的，因为时间序列数据具有时间依赖性，如果随机打乱，会破坏数据的时间顺序，从而影响模型学习到的时间依赖关系

#数据归一化
# 确保训练集和测试集的形状是二维数组
train = train.reshape(-1, 1)
test = test.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train) # 对训练数据进行拟合和转换
test_scaled = scaler.transform(test) # 使用同样的缩放参数转换测试数据

#重新创建时间序列数据
#训练集
X_train, y_train = [], []
for i in range(len(train_scaled) - config.timestep):
    # 从当前索引i开始，取sequence_length个连续的价格数据点，并将其作为特征添加到列表X_train中。
    X_train.append(train_scaled[i: i + config.timestep])
    # 将紧接着这sequence_length个时间点的下一个价格数据点作为目标添加到列表y_train中。
    y_train.append(train_scaled[i + config.timestep])
 
X_train = np.array(X_train)
#print(X_train)
y_train = np.array(y_train)
#print(y_train)
#测试集
X_test, y_test = [], []
for i in range(len(test_scaled) - config.timestep):
    X_test.append(test_scaled[i: i + config.timestep])
    y_test.append(test_scaled[i + config.timestep])
X_test, y_test = np.array(X_test), np.array(y_test)

#调整数据形状以适应LSTM的输入要求
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], config.feature_size))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], config.feature_size))

#定义LSTM网络
model = Sequential()

#添加LSTM层
model.add(LSTM(activation=config.activation, units=config.hidden_size, input_shape=(config.timestep, config.feature_size)))

#添加全连接层
model.add(Dense(config.output_size))

#编译LSTM模型
model.compile(optimizer='adam', loss='mean_squared_error')

#模型训练
history = model.fit(x=X_train, y=y_train, epochs=config.epochs, batch_size=config.batch_size, verbose=2)

#模型预测
predictions = model.predict(X_test)

#数据反向归一化
y_test_true_unnormalized = scaler.inverse_transform(y_test)
y_test_preds_unnormalized = scaler.inverse_transform(predictions)

#可视化
plt.figure(figsize=(10, 5))
plt.plot(y_test_true_unnormalized, label='True Values', marker='o')
plt.plot(y_test_preds_unnormalized, label='Predictions', marker='x')
plt.title('Comparison of True Values and Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Prices')
plt.legend()
plt.show()









































































