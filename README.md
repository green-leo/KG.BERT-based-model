# KG.EconomicProductKnowledgeGraph : Creating Product Knowledge Graph in Economic domain
The repository is modified from [KG-Bert](https://github.com/yao8839836/kg-bert) which is also modified from [pytorch-pretrained-BERT](https://github.com/huggingface/transformers).


## Technique
* Bert - BertForSequenceClassification


## Requirement
* torch
* pytorch_pretrained_bert
* ...

## Usage
The data was already modified for the training
* **congig.py** - change parameters if you want
* **dataset.py** - functions to tokenized and encoded data with pho-bert
* **engine.py** - train and eval functions
* **train.py** - run the file to train model
```bash
python train.py
```
