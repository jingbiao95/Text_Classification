# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
   Description :
   Author :       Jingbiao Li
   date：          2020/5/1
-------------------------------------------------
   Change Activity:
                   2020/5/1:
-------------------------------------------------
"""
from typing import List, Tuple
import tensorflow as tf


def bulid_data(dataset, data, config) -> List:
    """
    将数据转换为Model指定max_seq_len长度的序列
    :param dataset: 数据集的配置信息
    :param data:
    :param config:
    :return:
    """
    x, y, seq_len = data[0], data[1], data[2]
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=config.max_len)
    y = tf.keras.utils.to_categorical(y, num_classes=dataset.num_classes)
    return x, y
