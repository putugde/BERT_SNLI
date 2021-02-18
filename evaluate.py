import sys

import numpy as np

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn

# from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer
from transformers import RobertaTokenizer, DebertaTokenizer

# from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, AlbertForSequenceClassification
from transformers import RobertaForSequenceClassification, DebertaForSequenceClassification

from sklearn.metrics import f1_score

from utils import openData, removeMinVal

MAX_LENGTH = 128
batch_size = 512
label_dict = {
    'entailment':0,
    'neutral':1,
    'contradiction':2
}


def evaluate(model, device, dataloader, PARALLEL_GPU=False):

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader:
        
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
            # print('Loss are averaged! Assume that batch sizes are equally divided across the GPUs')
            loss = loss.mean()

        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader) 

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    total_pred = 0
    total_true = 0

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)} = {len(y_preds[y_preds==label])/len(y_true)}\n')
        total_pred += len(y_preds[y_preds==label])
        total_true += len(y_true)
    print(f'Unweighted accuracy : {total_pred}/{total_true} = {(total_pred/total_true)*100}%')

def main():
    if len(sys.argv) ==  3:
        FOLDER_NAME = sys.argv[1]
        EPOCH = sys.argv[2]
    else:
        print('ERROR : Please insert correct arguments! python evaluate.py <folder_name> <chosen_epoch>')
        return

    # Check GPU Availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU!')
    else:
        device = torch.device('cpu')
        print('Using CPU :(')

    # Import Dataset
    print('#### Importing Dataset ####')
    df_val = openData('./snli/dev.jsonl')
    df_test = openData('./snli/test.jsonl')

    # Preprocessing : removing data with label -1
    df_val = removeMinVal(df_val)
    df_test = removeMinVal(df_test)

    BERT_MODEL = 'microsoft/deberta-base'
    # BERT_MODEL = 'roberta-base'
    # BERT_MODEL = 'distilbert-base-uncased'
    # BERT_MODEL = 'bert-base-uncased'
    # BERT_MODEL = 'albert-base-v2'

    tokenizer = DebertaTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

    print('Encoding validation data')
    encode_val = tokenizer(df_val.premise.tolist(), df_val.hypothesis.tolist(), 
                        return_tensors='pt',padding='max_length', max_length = MAX_LENGTH)

    labels_val = torch.tensor(df_val.label.values)

    print('Encoding test data')
    encode_test = tokenizer(df_test.premise.tolist(), df_test.hypothesis.tolist(), 
                        return_tensors='pt',padding='max_length', max_length = MAX_LENGTH)

    labels_test = torch.tensor(df_test.label.values)

    dataset_val = TensorDataset(encode_val['input_ids'], encode_val['attention_mask'], labels_val)
    dataset_test = TensorDataset(encode_test['input_ids'], encode_test['attention_mask'], labels_test)

    model = DebertaForSequenceClassification.from_pretrained(BERT_MODEL,
                                                        num_labels=len(label_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False)
    
    PARALLEL_GPU = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        PARALLEL_GPU = True
    model = model.to(device)

    model.load_state_dict(torch.load(f'./models/{FOLDER_NAME}/finetuned_model_epoch_{EPOCH}.model', map_location=torch.device('cuda')))

    print('#### Validation Data Result ####')
    dataloader_validation = DataLoader(dataset_val, 
                                    sampler=SequentialSampler(dataset_val), 
                                    batch_size=batch_size)

    _, predictions, true_vals = evaluate(model, device, dataloader_validation, PARALLEL_GPU)
    accuracy_per_class(predictions, true_vals)

    print('#### Test Data Result ####')

    dataloader_test = DataLoader(dataset_test, 
                                   sampler=SequentialSampler(dataset_test), 
                                   batch_size=batch_size)

    _, predictions, true_vals = evaluate(model, device, dataloader_test, PARALLEL_GPU)
    accuracy_per_class(predictions, true_vals)

if __name__ == '__main__':
    main()

    