import json
import time
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForSequenceClassification 
import numpy as np
import pandas as pd
from scipy.special import softmax


def preprocess(text):
    new_text = []
    
    for t in text.split(" "):
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def get_labels():
    return ['negative', 'neutral', 'positive']

def get_model():
    return 'cardiffnlp/twitter-roberta-base-sentiment'

def get_sentiment(input_text, print_results=False, MODEL=get_model()):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    labels = ['negative', 'neutral', 'positive']
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    # model.save_pretrained(MODEL)
    # tokenizer.save_pretrained(MODEL)
    
    text = preprocess(input_text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    
    if print_results:
        print_data(scores, ranking, labels, text);
    
    return {'comment': input_text, 'scores': scores.tolist()}


def get_senitments(data, print_results=False, MODEL=get_model()):
        # Get all sentiments for a dataset
        # list(map(get_sentiment, comments))
    return {'model': MODEL, 'labels':get_labels(), 'results':[get_sentiment(comment, print_results, MODEL) for comment in data]}

def print_data(scores, ranking, labels, text):
    print(f'\n{text}')
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        print(f"{i+1}) {l} {np.round(float(s), 4)}")

def save_data(data, id, ext='json'):
    FILENAME = f'{id}-{int(time.time())}'
    
    match ext:
        case 'json':
            with open(f'./results/{FILENAME}.json','w+') as fp:
                json.dump(data, fp)
        case 'txt':
            pass
        case 'csv':
            pass
        case _:
            pass

# Converting the results obj to 
def to_csv(data):
    pass