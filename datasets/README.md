数据处理流程
# 超参数
```
tokenizer 分词器
vocabulary_size 词典大小
min_count 最小词频数

```
# 获取词典
1. 读取train数据，
2. 分词器分词，
3. 统计词频，
4. 对词频从大到小排序，获取vocabulary_size -1(其中1个是UNK)
5. 剔除小于min_count的词，获得最终的词典

