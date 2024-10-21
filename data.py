from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

def preprocess(text):
    new_text = []
    
    for t in text.split(" "):
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def get_labels():
    return ['negative', 'neutral', 'positive']

def get_sentiment(input_text, print_results=False, MODEL='cardiffnlp/twitter-roberta-base-sentiment'):
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
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            print(f"{i+1}) {l} {np.round(float(s), 4)}")
    
    return {'comment': input_text, 'scores': scores}
