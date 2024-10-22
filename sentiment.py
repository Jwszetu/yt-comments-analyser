import csv
import json
import re
import time
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForSequenceClassification 
import numpy as np
import pandas as pd
from scipy.special import softmax
from utils import ternary

class SentimentAnaylsis():
    def __init__(self):
        self._model = 'cardiffnlp/twitter-roberta-base-sentiment'
        self._labels = ['negative', 'neutral', 'positive']
    
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
    
    @property
    def labels(self):
        return self._labels
    @labels.setter
    def labels(self, l):
        self._labels = l

    def preprocess(self, text):
        new_text = []
        text = re.sub(r'(\t|\n)', '', text)
    
        for t in text.split(" "):
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def get_sentiment(self, input_text, print_results=False, model=None, labels=None):
        MODEL = ternary(model != None, model, self.model)
        LABELS = ternary(labels != None, labels, self.labels)
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        # model.save_pretrained(MODEL)
        # tokenizer.save_pretrained(MODEL)

        text = self.preprocess(input_text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        if print_results:
            self.print_data(scores, ranking, LABELS, text);

        return {'comment': text, 'scores': scores.tolist()}


    def get_senitments(self, data, print_results=False, model=None, labels=None):
        MODEL = ternary(model != None, model, self._model)
        LABELS = ternary(labels != None, labels, self._labels)
            # Get all sentiments for a dataset
            # list(map(get_sentiment, comments))
        results =  [self.get_sentiment(comment, print_results, MODEL, LABELS) for comment in data]
        scores = list(map(lambda a: a['scores'], results))

        return {
                'model': MODEL,
                'labels':LABELS,
                'stats': {'mean':  np.mean(scores, axis=0).tolist(), 'median': np.median(scores, axis=0).tolist()}, 
                'results': results}

    def print_data(self, scores, ranking, labels, text):
        print(f'\n{text}')
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            print(f"{i+1}) {l} {np.round(float(s), 4)}")

    def save_data(self, data, id, ext='json'):
        FILE_PATH = './results'
        FILENAME = f'{id}-{int(time.time())}'

        with open(f'{FILE_PATH}/{ext}/{FILENAME}.{ext}','w+') as fp:
            match ext:
                case 'json':
                        json.dump(data, fp)
                case 'csv':
                        csv_writer = csv.writer(fp, delimiter=',', quotechar='\'', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                        csv_writer.writerow(['model', 'comment', 'negative', 'neutral', 'positive'])
                        csv_writer.writerow([data['model'], 'mean'] + data['stats']['mean'])
                        csv_writer.writerow([data['model'], 'median'] + data['stats']['median'])

                        for result in data['results']:
                            csv_writer.writerow([data['model'], result['comment']] + result['scores'])
                case _:
                    pass
