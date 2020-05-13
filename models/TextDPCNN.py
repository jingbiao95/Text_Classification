# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     textcnn
   Description :
   Author :       Jingbiao Li
   date：          2020/5/13
-------------------------------------------------
   Change Activity:
                   2020/5/13:
-------------------------------------------------
"""
import os

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Flatten, Dense, Bidirectional


class Config():
    def __init__(self, dataset):
        # 数据集参数
        self.dataset = dataset

        # 预训练参数
        self.embeding_size = 300  # embedding维度大小

        # 模型参数
        self.model_name = "TextDPCNN"
        self.hidden_size = 128  # lstm   隐藏层个数
        self.num_layers = 2  # LSTM 层数
        self.num_classes = len(dataset.class_list)  # 类的数量
        self.epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.max_len = 32  # 每句话处理的长度
        self.lr = 0.001  # 学习率
        self.dropout = 0.5  # 随机失活
        # 模型训练结果
        self.save_path = os.path.join(self.dataset.data_dir, "saved_dict", self.model_name + ".h5")


class Model(tf.keras.Model):
    def __init__(self, config: Config):
        super(Model, self).__init__()

        self.config = config
        self.embeding = Embeding(input_dim=config.dataset.vocab_size, output_dim=config.embeding_size,
                                 input_length=config.max_len, weights=[config.dataset.embeding],
                                 trainable=False)
        self.biRNN = Bidirectional(LSTM(units=config.hidden_size, return_sequences=True, activation='relu'))
        self.dropout = Dropout(config.dropout)
        self.flatten = Flatten()
        self.out_put = Dense(units=config.num_classes, activation='softmax')

    def build(self, input_shape):
        super(Model, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        x_embeding = self.embeding(x)
        inputs = self.biRNN(inputs)
        inputs = self.dropout(inputs)
        inputs = self.flatten(inputs)
        out_put = self.out_put(inputs)
        return out_put
