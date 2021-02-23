import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset

# from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer
from transformers import RobertaTokenizer, DebertaTokenizer

# from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, AlbertForSequenceClassification
from transformers import RobertaForSequenceClassification, DebertaForSequenceClassification

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn

from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report

import random

import jsonlines

from utils import openData, removeMinVal

import logging

epochs = 10
MAX_LENGTH = 128
batch_size = 512
SEED_VAL = 10
LEARNING_RATE = 2e-5
EPS = 1e-8

# BERT_MODEL = 'microsoft/deberta-base'
BERT_MODEL = 'roberta-large'
# BERT_MODEL = 'distilbert-base-uncased'
# BERT_MODEL = 'bert-base-uncased'
# BERT_MODEL = 'albert-base-v2'

LOG_PATH = 'log/training-'+BERT_MODEL+'-'+str(epochs)+'-'+str(LEARNING_RATE)+'.log'
logging.basicConfig(filename=LOG_PATH, level=logging.INFO)

label_dict = {
    'entailment':0,
    'neutral':1,
    'contradiction':2
}

logging.info(' ######### Training Started! #########')

def initial_log():
    logging.info('\n ** Training Configuration **')
    logging.info(f'Model            : {BERT_MODEL}')
    logging.info(f'Epoch            : {epochs}')
    logging.info(f'Batch Size       : {batch_size}')
    logging.info(f'Learning Rate    : {LEARNING_RATE}')
    logging.info(f'EPS              : {EPS}')
    logging.info(f'Seed Value       : {SEED_VAL}\n')

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def evaluate(model, device, dataloader_val, PARALLEL_GPU=False):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]

        if PARALLEL_GPU:
            # Assumed that batch sizes are equally divided across the GPUs.
            loss = loss.mean()
        
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    # Hardcoded label
    target_names = ['entailment', 'neutral', 'contradiction']
    
    cr = classification_report(labels_flat, preds_flat, target_names=target_names, digits=4)
    logging.info(f'\n *** CLASSIFICATION REPORT ***\n\n{cr}\n')

    total_pred = 0
    total_true = 0
    total_acc = 0

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        logging.info(f'Class: {label_dict_inverse[label]}')
        logging.info(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)} = {len(y_preds[y_preds==label])/len(y_true)}\n')
        total_pred += len(y_preds[y_preds==label])
        total_true += len(y_true)
        total_acc += (total_pred/total_true)
    
    logging.info(f'Accuracy : {total_pred}/{total_true} = {(total_pred/total_true)}')
    # logging.info(f'Accuracy (equal weight each class): {total_acc/len(label_dict)} \n')

def main():
    initial_log()
    # Check GPU Availability
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Using GPU!')
        gpu_num = torch.cuda.device_count()
        print(f'There are {gpu_num} GPU available!')
        for i in range(0,gpu_num):
            gpu_name = torch.cuda.get_device_name(i)
            print(f' -- GPU {i+1}: {gpu_name}')
    else:
        device = torch.device('cpu')
        print('Using CPU :(')
    
    print('\n')

    # Import Dataset
    print('#### Importing Dataset ####')
    df_train = openData('./snli/train.jsonl')
    df_val = openData('./snli/dev.jsonl')
    df_test = openData('./snli/test.jsonl')

    print(df_train.shape)
    print(df_val.shape)
    print(df_test.shape)

    df_train = removeMinVal(df_train)
    df_val = removeMinVal(df_val)
    df_test = removeMinVal(df_test)

    print(df_train.shape)
    print(df_val.shape)
    print(df_test.shape)

    print('#### Download Tokenizer & Tokenizing ####')

    tokenizer = RobertaTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

    print('Encoding training data')
    encode_train = tokenizer(df_train.premise.tolist(), df_train.hypothesis.tolist(), 
                        return_tensors='pt',padding='max_length', max_length = MAX_LENGTH)

    labels_train = torch.tensor(df_train.label.values)

    print('Encoding validation data')
    encode_val = tokenizer(df_val.premise.tolist(), df_val.hypothesis.tolist(), 
                        return_tensors='pt',padding='max_length', max_length = MAX_LENGTH)

    labels_val = torch.tensor(df_val.label.values)

    print('Encoding test data')
    encode_test = tokenizer(df_test.premise.tolist(), df_test.hypothesis.tolist(), 
                        return_tensors='pt',padding='max_length', max_length = MAX_LENGTH)

    labels_test = torch.tensor(df_test.label.values)

    dataset_train = TensorDataset(encode_train['input_ids'], encode_train['attention_mask'], labels_train)
    dataset_val = TensorDataset(encode_val['input_ids'], encode_val['attention_mask'], labels_val)
    dataset_test = TensorDataset(encode_test['input_ids'], encode_test['attention_mask'], labels_test)

    print('#### Downloading Pretrained Model ####')
    model = RobertaForSequenceClassification.from_pretrained(BERT_MODEL,
                                                        num_labels=len(label_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False)

    dataloader_train = DataLoader(dataset_train, 
                                sampler=RandomSampler(dataset_train), 
                                batch_size=batch_size)

    dataloader_validation = DataLoader(dataset_val, 
                                    sampler=SequentialSampler(dataset_val), 
                                    batch_size=batch_size)

    dataloader_test = DataLoader(dataset_test, 
                                   sampler=SequentialSampler(dataset_test), 
                                   batch_size=batch_size)

    print('#### Setting Up Optimizer ####')
    optimizer = AdamW(model.parameters(),
                    lr=LEARNING_RATE, 
                    eps=EPS)

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train)*epochs)

    # TRAINING

    PARALLEL_GPU = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        PARALLEL_GPU = True
    model = model.to(device)


    random.seed(SEED_VAL)
    np.random.seed(SEED_VAL)
    torch.manual_seed(SEED_VAL)
    torch.cuda.manual_seed_all(SEED_VAL)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('#### Training Started! ####')
    for epoch in tqdm(range(1, epochs+1)):
        
        model.train()
        
        loss_train_total = 0

        # dataloader_train
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        i = 0
        for batch in progress_bar:
            i += 1
            model.zero_grad()
            
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }       

            outputs = model(**inputs)
            
            loss = outputs[0]

            if PARALLEL_GPU:
                # Assumed that batch sizes are equally divided across the GPUs.
                # print('Loss are averaged! Assume that batch sizes are equally divided across the GPUs')
                loss = loss.mean()

            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
            
        # Save model every epoch
        # torch.save(model.state_dict(), f'./models/deberta2/finetuned_model_epoch_{epoch}.model')
            
        tqdm.write(f'\nEpoch {epoch}')
        logging.info(f'\n -------- Epoch {epoch} ---------- \n')
        
        loss_train_avg = loss_train_total/len(dataloader_train)          
        tqdm.write(f'Training loss: {loss_train_avg}')
        logging.info(f'Training loss: {loss_train_avg}')
        
        val_loss, predictions, true_vals = evaluate(model, device, dataloader_validation, PARALLEL_GPU)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')
        logging.info(f'Validation loss: {val_loss}')
        logging.info(f'F1 Score (Weighted): {val_f1} \n \n')

        # Evaluate per epoch
        # Evaluate validation data
        logging.info(f' -- Validation Data -- \n')
        _, predictions, true_vals = evaluate(model, device, dataloader_validation, PARALLEL_GPU)
        accuracy_per_class(predictions, true_vals)

        logging.info(f' -- Test Data -- \n')
        # Evaluate test data
        _, predictions, true_vals = evaluate(model, device, dataloader_test, PARALLEL_GPU)
        accuracy_per_class(predictions, true_vals)

    # Save final epoch model
    torch.save(model.state_dict(), f'./models/roberta-large/finetuned_model_epoch_{epochs}.model')

if __name__ == '__main__':
    main()





