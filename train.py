from pickle import NONE
import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset

from ampligraph.evaluation import train_test_split_no_unseen

import config
import dataset
import engine


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {"acc": simple_accuracy(preds, labels)}


# feature output format - dictionary of:
# ids : tensor data_size x maxlen
# mask : tensor
# token_type_ids : tensor
# label : tensor

def process_data(train_path, test_path, val_path = None, label_list=config.LABELS):
    # print(train_path)
    train_df = pd.read_csv(train_path, dtype=object)
    train_feature = dataset.convert_to_features(train_df, label_list)
    train_data = TensorDataset(*(train_feature.values()))
    train_label = train_df['label'].to_numpy(dtype=int)
    train_size = len(train_df)

    test_df = pd.read_csv(test_path, dtype=object)
    test_feature = dataset.convert_to_features(test_df, label_list)
    test_data = TensorDataset(*(test_feature.values()))
    test_label = test_df['label'].to_numpy(dtype=int)
    test_size = len(test_df)

    if val_path is not None:
        val_df = pd.read_csv(val_path, dtype=object)
        val_feature = dataset.convert_to_features(val_df, label_list)
        val_data = TensorDataset(*(val_feature.values()))
        val_label = val_df['label'].to_numpy(dtype=int)
        val_size = len(val_df)
    else:
        val_data = NONE
        val_label = NONE
        val_size = NONE

    return train_data, train_label, train_size, \
        test_data, test_label, test_size, \
        val_data, val_label, val_size


if __name__ == "__main__":
    # prepare data
    print('Prepare Data....')
    label_list = config.LABELS
    num_labels = len(label_list)

    train_data, train_label, train_size, \
        test_data, test_label, test_size, \
        val_data, val_label, val_size = process_data(config.TRAIN_FILE_PATH, config.TEST_FILE_PATH)
    
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.TRAIN_BATCH_SIZE)

    val_sampler = SequentialSampler(test_data)
    val_dataloader = DataLoader(test_data, sampler=val_sampler, batch_size=config.VALID_BATCH_SIZE)


    # Prepare model
    print('Prepare Model....')
    device = torch.device(config.DEVICE)
    # model = EntityModel(num_tag=num_tag, num_pos=num_pos)
    model = BertForSequenceClassification.from_pretrained(config.BASE_MODEL_PATH, num_labels=num_labels)
    model.to(device)

    # Prepare optimizer
    print('Prepare Optimizer....')
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # optimizer_parameters = [
    #     {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.001,},
    #     {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
    # ]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


    num_train_optimization_steps = int(
            train_size / config.TRAIN_BATCH_SIZE / config.GRADIENT_ACCUMULATION_STEPS) * config.EPOCHS

    num_train_steps = int(train_size / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    # optimizer = AdamW(optimizer_parameters, lr=3e-5)
    optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=config.LEARNING_RATE,
                             warmup=config.WARMUP_PROPORTION,
                             t_total=num_train_optimization_steps)


    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )



    # Training
    print('Training....')

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        # train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        train_loss = engine.train_fn(train_dataloader, model, optimizer, device)
        test_loss, preds = engine.eval_fn(val_dataloader, model, device)
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, val_label.numpy())

        print(f"Train Loss = {train_loss} Valid Loss = {test_loss} ACC = {result['acc']}")
        if test_loss < best_loss:
            # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model.state_dict(), config.MODEL_PATH)
            model.config.to_json_file(config.CONFIG_PATH)
            best_loss = test_loss
