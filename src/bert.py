from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.autograd import Variable
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import *
from torch import nn
import numpy as np
import time
import datetime
import torch
import random
import json
import os
import sys
import torch.nn.functional as F
import unicodedata
import re

use_gpu = True
seed = 1234
batch_size = 64 # 8 per available GPU
max_length = 512
folds = 10
n_epochs = 10
lr = 2e-5
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
#device_ids = [0, 1, 2, 3]

if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

tokenizer = None

def prepare_set(dataset, max_length=256):
    """returns input_ids, input_masks, labels for set of data ready in BERT format"""
    global tokenizer
    input_ids, labels = [], []
    
    for i in dataset:
        input_ids.append(i["text"].lower())
        labels.append(1 if i["label"] == 1 else 0)
    
    inputs = [ tokenizer.encode_plus(i, pad_to_max_length=True, add_special_tokens=True, max_length=max_length) for i in input_ids ]
    input_ids = [ torch.tensor(x["input_ids"]) for x in inputs ] 
    input_masks = [ torch.tensor(x["attention_mask"]) for x in inputs ] 
    token_type_ids = [ torch.tensor(x["token_type_ids"]) for x in inputs ] 

    input_ids = torch.stack(input_ids)
    input_masks = torch.stack(input_masks)
    token_type_ids = torch.stack(token_type_ids)
    labels = torch.tensor(labels)

    return (input_ids, input_masks, token_type_ids), labels


def predict(self, test):
    (test_inputs, test_masks, test_type_ids), y_test = prepare_set(test, max_length=max_length)
    test_data = TensorDataset(test_inputs, test_masks, test_type_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    self.eval()
    with torch.no_grad(): 
        preds = []
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_token_type_ids = tuple(t.to(device) for t in batch)
            output = self(b_input_ids, b_input_mask, b_token_type_ids)
            logits = output[0].detach().cpu().numpy()
            preds += list(np.argmax(logits, axis=1).flatten())

    return preds


def get_bert(pretrained_model):
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=2)

    # if torch.cuda.device_count() > 1 and device.type == "cuda":
    #     model = nn.DataParallel(model, device_ids=device_ids)

    model.to(device)
    model.predict = predict.__get__(model)

    return model


def build_bert(train, dev, pretrained_model, n_epochs=15):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
    model_path = "temp.pt"

    print([len(x) for x in (train, dev)])

    (train_inputs, train_masks, train_type_ids), y_train = prepare_set(train, max_length=max_length)
    train_data = TensorDataset(train_inputs, train_masks, train_type_ids, y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our dev set.
    (dev_inputs, dev_masks, dev_type_ids), y_dev = prepare_set(dev, max_length=max_length)
    dev_data = TensorDataset(dev_inputs, dev_masks, dev_type_ids, y_dev)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    total_steps = len(train_dataloader) * n_epochs
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                        num_warmup_steps = 0,
                                        num_training_steps = total_steps)

    model.zero_grad()
    best_score = 0
    best_loss = 1e6
    train_losses = []

    for epoch in range(n_epochs):

        start_time = time.time()
        train_loss = 0 
        model.train()

        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_token_type_ids, b_labels = tuple(t.to(device) for t in batch)
            output = model(b_input_ids, 
                            attention_mask=b_input_mask,
                            token_type_ids=b_token_type_ids,
                            labels=b_labels)


            loss = output[0].sum()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            model.zero_grad()

        scheduler.step()
        elapsed = time.time() - start_time
        model.eval()
        val_preds = []
        with torch.no_grad(): 
            val_loss, batch = 0, 1
            for batch in dev_dataloader:
                b_input_ids, b_input_mask, b_token_type_ids, b_labels = tuple(t.to(device) for t in batch)
                output = model(b_input_ids, 
                            attention_mask=b_input_mask,
                            token_type_ids=b_token_type_ids,
                            labels=b_labels)
                
                loss = output[0].sum()
                val_loss += loss.item()
                logits = output[1].detach().cpu().numpy()
                val_preds += list(np.argmax(logits, axis=1).flatten())
                model.zero_grad()

        val_score = f1_score(y_dev.cpu().numpy().tolist(), val_preds, average="macro")
        #print("Epoch %d Train loss: %.4f. Validation F1-Score: %.4f  Validation loss: %.4f. Elapsed time: %.2fs."% (epoch + 1, train_loss, val_score, val_loss, elapsed))

        if val_score > best_score:
            torch.save(model.state_dict(), model_path)
            #print(classification_report(y_dev.cpu().numpy().tolist(), val_preds, digits=4))
            best_score = val_score

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.predict = predict.__get__(model)
    os.remove(model_path)

    return model



