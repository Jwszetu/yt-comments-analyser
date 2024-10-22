import sys
from dotenv import load_dotenv
from client import get_comments
from sentiment import SentimentAnaylsis as senti
import argparse
import os
import re

def main(): 

    # Generates the parser object to read CLI arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # Define the regex pattern to extract the video if from the youtube URL
    url_pattern = r'\?v=([\S]+),?'
        
    # Loading .env file
    load_dotenv()
    
    # Assigning the video Id based on the arguments recieved
    if(args.u):
        match = re.search(url_pattern, args.u)
        if match:
            video_id = match.group(1)
    elif (args.i):
        video_id = args.i
    
    if(args.n):
        n_Results = args.n | 1

    # Fetches the yt comments from a video
    comments = get_comments(os.getenv("API_KEY"), video_id, n_Results) 
    
    err_exit(len(comments) == 0, 'No comments found.')
    
    sentiAnalysis = senti()
    
    results = sentiAnalysis.get_senitments(comments, print_results=args.p)
    
    if(args.o):
        sentiAnalysis.save_data(results, video_id, args.o)


def setup_parser():
    parser = argparse.ArgumentParser(description='Youtube Comment Sentiment Analyzer')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-u', help='youtube video url', nargs='?')
    group.add_argument('-id', help='youtube video id', nargs='?')
    parser.add_argument('-n', help='number of results',  nargs='?', type=int, metavar='N', default=5, choices=range(1,50))
    # print output
    parser.add_argument('-p', help='print output', action='store_true')
    # Output Format .json, .txt, .csv
    parser.add_argument('-o', help='save result to output', nargs='?', choices=['csv', 'json'])
    return parser

def err_exit(test=False, msg = ''):
    if test:
        print(f'Error: {msg}')
        sys.exit(1)

# boilerplate
if __name__ == "__main__":
    main()