import os

import numpy as np
import pandas as pd
from bert4keras.backend import keras, set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Lambda, Dense

set_gelu('tanh')  # 切换gelu版本

CONFIG = {
    'max_len': 32,
    'batch_size': 64,
    'epochs': 32,
    'use_multiprocessing': True,
    'model_dir': os.path.join('model_file'),
}
num_classes = 5
max_len = 64
batch_size = 32
config_path = 'electra_tiny/bert_config_tiny.json'
checkpoint_path = 'electra_tiny/model.ckpt-1000000'
dict_path = 'electra_tiny/vocab.txt'

five_CLS_data = pd.read_csv("data/five_CLS_data.csv").values.tolist()
label_id = {}


def split(data):
    train_data_split, valid_data_split, test_data_split = [], [], []
    class_index = 0
    for index, message in enumerate(data):
        if message[1] not in label_id.keys():
            label_id[message[1]] = class_index
            class_index += 1
        if index % 5 != 1 and index % 5 != 2:
            train_data_split.append([message[0], label_id[message[1]]])
        elif index % 5 == 1:
            valid_data_split.append([message[0], label_id[message[1]]])
        else:
            test_data_split.append([message[0], label_id[message[1]]])
    return train_data_split, valid_data_split, test_data_split


train_data, valid_data, test_data = split(five_CLS_data)
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            if isinstance(text, str):
                token_ids, segment_ids = tokenizer.encode(text, max_length=max_len)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='electra',
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dense(units=num_classes,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()
AdamLR = extend_with_piecewise_linear_lr(Adam)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=AdamLR(learning_rate=1e-4,
                     lr_schedule={1000: 1, 2000: 0.1}),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
              (val_acc, self.best_val_acc, test_acc))


evaluator = Evaluator()
save = ModelCheckpoint(
    os.path.join(CONFIG['model_dir'], 'bert.h5'),
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='auto'
)
early_stopping = EarlyStopping(
    monitor='loss',
    min_delta=0,
    patience=8,
    verbose=1,
    mode='auto'
)
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=10,
                    callbacks=[evaluator, save, early_stopping])

model.load_weights('best_model.weights')
print(u'final test acc: %05f\n' % (evaluate(test_generator)))
print(u'final test acc: %05f\n' % (evaluate(valid_generator)))
