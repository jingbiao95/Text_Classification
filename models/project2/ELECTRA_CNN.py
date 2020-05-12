import os
import kashgari
from kashgari.embeddings import BERTEmbeddingV2
from kashgari.tokenizer import BertTokenizer
import pandas as pd
import keras
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping

model_folder = 'electra_tiny'

checkpoint_path = os.path.join(model_folder, 'model.ckpt-1000000')
config_path = os.path.join(model_folder, 'bert_config_tiny.json')
vocab_path = os.path.join(model_folder, 'vocab.txt')


def data_doubles(train_data):
    train_features = [list(i[0]) for i in train_data]
    train_labels = [i[1] for i in train_data]
    return train_features, train_labels


def read_message(path):
    data = pd.read_csv(path).values.tolist()
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    train_data, valid_data, test_data = [], [], []
    for i, j in enumerate(random_order):
        if i % 10 != 0 and i % 10 != 1:
            train_data.append(data[i])
        if i % 10 == 0:
            valid_data.append(data[i])
        if i % 10 == 1:
            test_data.append(data[i])
    train_features, train_labels = data_doubles(train_data)
    valid_features, valid_labels = data_doubles(valid_data)
    test_features, test_labels = data_doubles(test_data)
    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels


train_features, train_labels, valid_features, valid_labels, test_features, test_labels = read_message(
    "data/five_CLS_data.csv")

feature_list = []
label_list = []
max_len = 128
# 根据数据集中的最大信息长度判断当前任务的max-len
# for i in five_CLS_data:
#     feature_list.append(i[0])
#     if len(i[0]) > max_len:
#         max_len = len(i[0])
#     label_list.append(i[1])
tokenizer = BertTokenizer.load_from_vocab_file(vocab_path)
embed = BERTEmbeddingV2(vocab_path,
                        config_path,
                        checkpoint_path,
                        bert_type='electra',
                        task=kashgari.CLASSIFICATION,
                        sequence_length=max_len)

tf_board_callback = keras.callbacks.TensorBoard(log_dir='tf_dir', update_freq=10)

from kashgari.tasks.classification import CNNLSTMModel, CNNModel

save = ModelCheckpoint(
    os.path.join('model_dir', 'CNNModel_bert.h5'),
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    mode='auto'
)
early_stopping = EarlyStopping(
    monitor='val_acc',
    min_delta=0,
    patience=8,
    verbose=1,
    mode='auto'
)
model = CNNModel(embed)

# ------------ build model ------------
model.fit(
    train_features, train_labels,
    valid_features, valid_labels,
    epochs=60,
    batch_size=256,
    callbacks=[tf_board_callback, save, early_stopping]
)
model.evaluate(test_features, test_labels)
