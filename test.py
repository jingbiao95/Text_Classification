# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test
   Description :
   Author :       Jingbiao Li
   date：          2020/5/1
-------------------------------------------------
   Change Activity:
                   2020/5/1:
-------------------------------------------------
"""
import argparse
from importlib import import_module

import utils

parse = argparse.ArgumentParser(description="文本分类训练主参数设置")
parse.add_argument("--dataset", default="ten_news_data")
parse.add_argument("--model", default="TextRNN", help="文本分类模型")
args = parse.parse_args()
if __name__ == '__main__':
    # 加载数据集
    x = import_module('processing.' + args.dataset)
    dataset = x.DataSet()

    # 加载模型
    x = import_module("models." + args.model)
    config = x.Config(dataset)
    model = x.Model(config)

    # 准备训练
    test_x, test_y = utils.bulid_data(dataset, dataset.test, config)

    model.build(input_shape=(None, config.max_len))
    model.summary()
    model.load_weights(config.save_path)
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    history = model.evaluate(test_x, test_y, verbose=2)
    print("Test Score:", history[0])
    print("Test Accuracy:", history[1])
