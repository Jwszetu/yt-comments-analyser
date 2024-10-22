# Youtube Comment Sentiment Analysis CLI Tool

small cli tool that pulls comments using the youtube api then does sentiment analysis based on
a pretrained model (cardiffnlp/twitter-roberta-base-sentiment).

full options list below 

## Usage

``` 
    usage: main.py [-h] [-u [U] | -id [ID]] [-n [N]] [-p] [-o [{csv,json}]]

    Youtube Comment Sentiment Analyzer

    options:
      -h, --help       show this help message and exit
      -u [U]           youtube video url
      -id [ID]         youtube video id
      -n [N]           number of results
      -p               print output
      -o [{csv,json}]  save result to output
```