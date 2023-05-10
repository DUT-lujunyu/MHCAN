# Multi-task Hierarchical Cross-Attention Network for Multi-label Text Classification

*Code for NLPCC2022 Task5 Track1 "Multi-label Classification Model for English Scientific Literature".*

Multi-label Classification Model for English Scientific Literature: develop a multi-label classification model for scientific research literature based on the given metadata (title and abstract) of scientific research literature and corresponding hierarchical labels in a specific domain.

## Requirements

~~~
python 3.8
transformers 3.4.0
pytorch 1.10.0+cu113
numpy 1.21.4
tqdm
~~~

## Data

You can reconfigure settings in **config.py**. 

Original dataset: 

```python
"train_data" : 'data/training set_modified.json'
"dev_data" : 'data/verification set_modified.json'
```

Provide a subset for testing: 

```python
"train_data" : 'data/test_training data.json'    #  (2000)
"dev_data" : 'data/test_verification data.json'  #  (500)
```

## Running

```shell
python run.py --embedding [bert/glove] --model [NN/HAM] --start_train [True/False]
```

## Results

The training results are saved in two files: [Modelname] \_all\_scores.txt  [Modelname]\_top5.txt

[Modelname] \_all\_scores.txt :  The training set score and the validation set score for each epoch (including loss)

[Modelname]\_top5\_[scores\_key].txt :  The model's best five validation set scores (including loss)

## Description:

Version: 1.1
Author: Lu Junyu, Zhang hao
Date: 2022-04-19 15:26:12
LastEditors: Lu Junyu
LastEditTime: 2022-07-29 15:26:12\

## Cite
~~~
@inproceedings{lu2022multi,
  title={Multi-task Hierarchical Cross-Attention Network for Multi-label Text Classification},
  author={Lu, Junyu and Zhang, Hao and Shen, Zhexu and Shi, Kaiyuan and Yang, Liang and Xu, Bo and Zhang, Shaowu and Lin, Hongfei},
  booktitle={Natural Language Processing and Chinese Computing: 11th CCF International Conference, NLPCC 2022, Guilin, China, September 24--25, 2022, Proceedings, Part II},
  pages={156--167},
  year={2022},
  organization={Springer}
}
~~~

