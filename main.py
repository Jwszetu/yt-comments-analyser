from dotenv import load_dotenv
from data import get_sentiment
import requests
import argparse
import os
import re

def main(): 
    parser = argparse.ArgumentParser(description='Youtube Comment Sentiment Analyzer')
    parser.add_argument('-u', help='youtube video url', nargs='?')
    parser.add_argument('-id', help='youtube video id', nargs='?')
    parser.add_argument('-n', help='number of results', nargs='?', type=int, default=5)
    
    
    args = parser.parse_args()
    url_pattern = r'\?v=([\S]+),?'
    video_id = ''
    n_Results = 0
    
    load_dotenv()
    
    # print(args)
    
    if(args.u):
        match = re.search(url_pattern, args.u)
        if match:
            video_id = match.group(1)
    elif (args.i):
        video_id = args.i
    
    if(args.n):
        n_Results = args.n


    url = f'https://www.googleapis.com/youtube/v3/commentThreads?key={os.getenv("API_KEY")}&textFormat=plainText&part=snippet&videoId={video_id}&maxResults={n_Results}'
    response = requests.get(url)
    data = response.json()['items']

    comments = list(map(getText, data))
    
    #print(comments)
    
    results = []
    
    for comment in comments:
        results.append(get_sentiment(comment))

def getText(comment):
    return comment['snippet']['topLevelComment']['snippet']['textOriginal']

# boilerplate
if __name__ == "__main__":
    main()