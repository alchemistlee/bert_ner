## 已支持如下实体
[ENTITY(股票), DATE(日期), NUMBER(数字), METRIC(指标)]

## 单数据源  
```
CUDA_VISIBLE_DEVICES=1 python train1.py --logdir checkpoints/merge_data --batch_size 32 --top_rnns --lr 1e-3 --n_epochs  300  --trainset_1 jiashi/merge_train.txt --validset jiashi/merge_valid.txt 
```

## Ensemble Model
```
CUDA_VISIBLE_DEVICES=1 python train.py --logdir checkpoints/merge_data --batch_size 32 --top_rnns --lr 1e-3 --n_epochs  300  --trainset_1 jiashi/merge_train.txt --validset jiashi/merge_valid.txt 
```

## 预估服务flask
```
python infer.py
```

## 标注数据格式化为BIO
POST接口：ip:8088/dump/brat_batch
```
BODY: 
{
    "reqFilename": "/Users/zhangyong/Documents/algorithm/train_data/hx_20190624.txt",
    "annFileRootPath": "/Users/zhangyong/Documents/algorithm/train_data/0624/"
}
```
RELEASE分支
JAVA github: https://github.com/Tigerye/jiashi-ner

### 数据格式
```
还 O
有 O
很 O
大 O
的 O
提 O
升 O
空 O
间 O
, O
目 B-DATE
前 I-DATE
中 O
国 O
每 B-NUMBER
个 I-NUMBER
订 B-METRIC
阅 I-METRIC
会 I-METRIC
员 I-METRIC
平 O
均 O
拥 O
有 O
1.2 B-NUMBER
个 I-NUMBER
视 O
频 O
网 O
站 O
账 O
户 O
, O
美 O
国 O
平 O
均 O
为 O
2.5 B-NUMBER
个 I-NUMBER
; O
```
