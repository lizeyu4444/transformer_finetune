# 利用huggingface transformer进行迁移学习

通常情况下，对于个人或者小企业来说，缺乏训练集是一个令人头痛的问题。huggingceface开源了基于transformer的模型库，集成了众多架构，例如Bert、GPT、GPT-2、Transformer-XL等，并且能够让开发者上传分享模型。利用训练好的模型，进行迁移学习，能够大大提高训练效率和改善结果。本仓库利用了`multilingual_sentiment_vocab20k`模型进行finetune，如果需要预训练模型可以去[huggingface官网](https://huggingface.co/models)下载，下载好之后，参考scripts/中的notebook进行训练和预测。

## 使用与安装

1. 安装依赖

2. 数据预处理

参考`scripts/data_utils.py`，生成指定的数据格式。可参考`data/processed/train.tsv`

3. 训练模型
```
# 修改train.py中的参数
python train.py
```

4. 预测数据集
```
# 修改文件路径
python predict.py
```

4. 模型部署，略
```
```

## 项目结构
```
├── data                                  // 处理过后的数据
├── train.py                              // 训练入口
├── predict.py                            // 预测入口
├── model.py                              // 模型类
├── processors.py                         // 数据处理类
├── arguments.py                          // 参数类
├── scripts                               // 脚本文件
│   ├── data_utils.ipynb                  // 数据预处理，生成模型所需格式
│   ├── load_pretrained_models.ipynb      // 导入训练好的模型，进行预测
│   └── update_vocab.py                   // 利用新数据的词汇表，填充原有预训练模型中的无用词汇，可以不管
│
└── regex                                 // 正则文件
```

