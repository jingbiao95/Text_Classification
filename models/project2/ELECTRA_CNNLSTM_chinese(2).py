import os
import os
import kashgari
from kashgari.embeddings.bert_embedding_v2 import BERTEmbeddingV2
from kashgari.tokenizer import BertTokenizer
import pandas as pd

model_folder = 'electra_tiny'

checkpoint_path = os.path.join(model_folder, 'model.ckpt-1000000')
config_path = os.path.join(model_folder, 'bert_config_tiny.json')
vocab_path = os.path.join(model_folder, 'vocab.txt')

five_CLS_data = pd.read_csv("data/five_CLS_data.csv").values.tolist()

feature_list = []
label_list = []
max_len = 0
for i in five_CLS_data:
    feature_list.append(i[0])
    if len(i[0]) > max_len:
        max_len = len(i[0])
    label_list.append(i[1])
tokenizer = BertTokenizer.load_from_vacab_file(vocab_path)
embed = BERTEmbeddingV2(vocab_path,
                        config_path,
                        checkpoint_path,
                        bert_type='electra',
                        task=kashgari.CLASSIFICATION,
                        sequence_length=max_len)


# bert分词
sentences_tokenized = [tokenizer.tokenize(s) for s in feature_list]
print(sentences_tokenized)

train_x, train_y = sentences_tokenized[:2], label_list[:2]
validate_x, validate_y = sentences_tokenized[2:], label_list[2:]

from kashgari.tasks.classification import CNNLSTMModel

model = CNNLSTMModel(embed)

# ------------ build model ------------
model.fit(
    train_x, train_y,
    validate_x, validate_y,
    epochs=3,
    batch_size=32
)
