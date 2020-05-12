import numpy as np
import pandas as pd
from bert4keras.backend import keras, set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
from keras.layers import Lambda, Dense

set_gelu('tanh')  # 切换gelu版本

config_path = '../electra_tiny/bert_config_tiny.json'
checkpoint_path = '../electra_tiny/model.ckpt-1000000'
dict_path = '../electra_tiny/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
# 加载预训练模型


max_len = 128
batch_size = 32


def read_message(path):
    features_data = pd.read_csv(path).values.tolist()
    label_counts_number = 0
    label_id_dict = {}
    data = []
    train_data_split, valid_data_split, test_data_split = [], [], []
    for i in features_data:
        if i[1] not in label_id_dict.keys():
            label_id_dict[i[1]] = label_counts_number
            label_counts_number += 1
        if i[0] is not None:
            data.append([i[0], label_id_dict[i[1]]])
    random_order = list(range(len(features_data)))
    np.random.shuffle(random_order)
    for index, message in enumerate(random_order):
        message = features_data[message]
        if message[0] is not None:
            if index % 5 != 1 and index % 5 != 2:
                train_data_split.append([message[0], label_id_dict[message[1]]])
            elif index % 5 == 1:
                valid_data_split.append([message[0], label_id_dict[message[1]]])
            else:
                test_data_split.append([message[0], label_id_dict[message[1]]])

    return label_counts_number, train_data_split, valid_data_split, test_data_split
    # 训练模型


# 加载数据集
label_counts, train_data, valid_data, test_data = read_message('train.csv')


# 建立分词器


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

)  # 建立模型，加载权重

output = Lambda(lambda x: x[:, 0],
                name='CLS-token')(bert.model.output)
output = Dense(units=label_counts,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(lr=1e-4),
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
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=10,
                    callbacks=[evaluator])

model.load_weights('best_model.weights')
print(u'final test acc: %05f\n' % (evaluate(test_generator)))
