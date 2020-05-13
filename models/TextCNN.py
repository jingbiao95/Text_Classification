# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     textcnn
   Description :
   Author :       Jingbiao Li
   date：          2020/5/1
-------------------------------------------------
   Change Activity:
                   2020/5/1:
-------------------------------------------------
"""
import os

import tensorflow as tf
import tensorflow.keras as keras


class Config():
    def __init__(self, dataset):
        # 数据集参数
        self.dataset = dataset

        # 预训练参数
        self.embeding_size = 300  # embedding维度大小

        # 模型参数
        self.model_name = "TextCNN"
        self.convs = [3, 4, 5]  # 卷积
        self.num_classes = len(dataset.class_list)  # 类的数量
        self.epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.max_len = 32  # 每句话处理的长度
        self.lr = 0.001  # 学习率
        self.num_filter = 256  # 卷积核数量(channels数)
        self.dropout = 0.5  # 随机失活
        # 模型训练结果
        self.save_path = os.path.join(self.dataset.data_dir, "saved_dict", self.model_name + ".h5")


class Model(tf.keras.Model):
    """
    TextCNN模型
    """

    def __init__(self, config):
        super(Model,self).__init__()
        self.config = config
        self.embedding = keras.layers.Embedding(config.dataset.vocab_size, config.embeding_size,
                                                input_length=config.max_len, weights=[config.dataset.embeding],
                                                trainable=False)
        self.convs = [keras.layers.Conv1D(config.num_filter, kernel_size, strides=1, padding="same", activation='relu')
                      for kernel_size in config.convs]
        self.maxpooling = tf.keras.layers.MaxPool1D()
        self.flattern = tf.keras.layers.Flatten()
        self.dropouts = tf.keras.layers.Dropout(config.dropout)
        self.out = tf.keras.layers.Dense(config.num_classes, activation="softmax")

    def conv_and_poll(self, x, conv):
        x = conv(x)
        x = self.maxpooling(x)
        return x

    def build(self, input_shape):
        super(Model, self).build(input_shape)

    def call(self, x):
        x = self.embedding(x)
        x = tf.keras.layers.concatenate([self.conv_and_poll(x, conv) for conv in self.convs], axis=-1)
        x = self.flattern(x)
        x = self.dropouts(x)
        x = self.out(x)
        return x
