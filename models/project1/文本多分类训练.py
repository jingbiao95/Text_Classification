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
num_classes = 3
maxlen = 64
batch_size = 32
config_path = 'albert_tiny_google_zh_489k/albert_config.json'
checkpoint_path = 'albert_tiny_google_zh_489k/albert_model.ckpt'
dict_path = 'albert_tiny_google_zh_489k/vocab.txt'

one = pd.read_csv("data/xywy.尿毒症.csv").question_type.values.tolist()
two = pd.read_csv("data/xywy.帕金森.csv").question_type.values.tolist()
three = pd.read_csv("data/xywy.心血管内科.csv").question_type.values.tolist()
all_train_data = []
all_valid_data = []
all_test_data = []


def random_split(data):
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    train_data = [data[i] for i, j in enumerate(random_order) if i % 10 != 0 and i % 10 != 1]
    valid_data = [data[i] for i, j in enumerate(random_order) if i % 10 == 0]
    test_data = [data[i] for i, j in enumerate(random_order) if i % 10 == 1]
    return train_data, valid_data, test_data


train_one, valid_one, test_one = random_split(one)
train_two, valid_two, test_two = random_split(two)
train_three, valid_three, test_three = random_split(three)


def append_train_label(train_data, label):
    for i in train_data:
        all_train_data.append([i, label])


def append_valid_label(valid_data, label):
    for i in valid_data:
        all_valid_data.append([i, label])


def append_test_label(test_data, label):
    for i in test_data:
        all_test_data.append([i, label])


append_train_label(train_one, 0)
append_train_label(train_two, 1)
append_train_label(train_three, 2)
append_valid_label(valid_one, 0)
append_valid_label(valid_two, 1)
append_valid_label(valid_three, 2)
append_test_label(test_one, 0)
append_test_label(test_two, 1)
append_test_label(test_three, 2)
train_data = all_train_data
valid_data = all_valid_data
test_data = all_test_data

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            if isinstance(text,str):
                token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
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
    model='albert',
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
    # optimizer=Adam(1e-5),  # 用足够小的学习率
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
