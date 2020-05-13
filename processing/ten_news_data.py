# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
   Description :
   Author :       Jingbiao Li
   date：          2020/5/5
-------------------------------------------------
   Change Activity:
                   2020/5/5:
-------------------------------------------------
"""
import os
from collections import Counter
import pickle as pkl
import numpy as np

UNK, PAD = "<UNK>", "<PAD>"  # 未知字，padding符号
# 词嵌入路径
sogou_embed_path = "datasets/embed/sgns.sogou.char"


def build_embeding(data_name, vocab, embed_path):
    if os.path.exists():
        pass
    with open(embed_path, "r", encoding="utf8") as f:
        for line in f.readlines():
            line = line.strip().split(" ")
            if line[0] in word_to_id:
                idx = word_to_id[line[0]]
                emb = [float(x) for x in line[1:301]]
                embeddings[idx] = np.asarray(emb, dtype='float32')
        pass


class DataSet():
    def __init__(self):
        # 数据集基本信息
        self.data_name = "ten_news_data"
        self.data_dir = os.path.join("datasets", self.data_name)
        self.tokenizer = lambda x: [_ for _ in x]  # 分词器
        self.train_path = os.path.join(self.data_dir, 'data', "train.txt")
        self.test_path = os.path.join(self.data_dir, 'data', "test.txt")
        self.dev_path = os.path.join(self.data_dir, 'data', 'dev.txt')
        self.class_list = [line.strip() for line in
                           open(os.path.join(self.data_dir, "data", 'class.txt'), "r", encoding='utf8').readlines()]
        self.num_classes = len(self.class_list)

        # 数据集超参数
        self.vocab_size = 10000  # 词表长度限制
        self.min_count = 1  # 词频最小次数

        self.vocab = self.build_vocab()  # 获得词表(有次序的)
        self.sogo_embeding_path = "datasets/embed/sgns.sogou.char"
        self.embeding_path = os.path.join(self.data_dir, 'data', "embedding_SougouNews")
        self.embeding_size = 300
        self.embeding = self.load_embeding()  # 加载词向量
        # 开始加载数据
        self.train = self.read_data(self.train_path)
        self.test = self.read_data(self.test_path)
        self.dev = self.read_data(self.dev_path)

    def build_vocab(self):
        """
        根据vocabulary_size,和min_count 生成词表
        :return:
        """
        vacab_path = os.path.join(self.data_dir, "data",
                                  "vocab_" + str(self.vocab_size) + "_" + str(self.min_count) + ".pkl")
        if os.path.exists(vacab_path):
            return pkl.load(open(vacab_path, 'rb'))
        else:
            word_count = {}  # {word:count} 词：词频
            with open(self.train_path, 'r', encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    content = line.split("\t")[0]
                    for word in self.tokenizer(content):
                        word_count[word] = word_count.get(word, 0) + 1
            word_count = {key: count for key, count in word_count.items() if
                          count >= self.min_count}  # 剔除词频低于min_count
            word_count = sorted(word_count, key=word_count.get, reverse=True)[:self.vocab_size - 1]
            vocab = {word_count[0]: idx + 2 for idx, word_count in enumerate(word_count)}
            vocab.update({PAD: 0})
            vocab.update({UNK: 1})
            pkl.dump(vocab, open(vacab_path, 'wb', ))
            return vocab

    def read_data(self, file_path):
        """
        将数据读入内存中
        :param file_path:
        :return:
        """
        x, y, seq_len = [], [], []  # x,y,x的长度
        with open(file_path, "r", encoding="utf-8")as f:
            for line in f.readlines():
                [content, label] = line.split("\t")
                word_list = self.tokenizer(content)
                id_list = [self.vocab.get(word, 0) for word in word_list]
                seq_len.append(len(id_list))
                x.append(id_list)
                y.append(label)

        return (x, y, seq_len)

    def load_embeding(self):
        if os.path.exists(self.embeding_path):
            embeding = np.load(self.embeding_path)
        else:
            # 声明全0
            embeding = np.zeros([self.vocab_size, self.embeding_size],dtype=np.float64)
            # 读取
            vocab = self.vocab.keys()
            with open(self.sogo_embeding_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line_split = line.strip().split(" ")
                    if line_split[0] in vocab:
                        embeding[self.vocab.get(line_split[0])] = np.asarray(line_split[1:], dtype=np.float64)
            np.savez_compressed(self.embeding_path, embeding=embeding)
        return embeding
