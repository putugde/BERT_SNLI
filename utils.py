import jsonlines
import pandas as pd

def mapToNumber(gold_label):
    if gold_label == 'entailment':
        return 0
    if gold_label == 'neutral':
        return 1
    if gold_label == 'contradiction':
        return 2
    return -1

def openData(path):
    data = []
    i = 0

    with jsonlines.open(path) as f:

        for line in f.iter():  
            data.append({
                'premise':line['sentence1'],
                'hypothesis':line['sentence2'],
                'label':mapToNumber(line['gold_label'])
            })

    df = pd.DataFrame(data)
    return df

def removeMinVal(df):
    # print(df['label'].unique())
    # Remove data with label -1 (undefined)
    new_df = df[df.label != -1]
    new_df.reset_index(drop=True, inplace=True)
    # print(new_df['label'].unique())
    return new_df