
import requests

def get_comments(API_KEY, video_id, num_Results):
    url = f'https://www.googleapis.com/youtube/v3/commentThreads?key={API_KEY}&textFormat=plainText&part=snippet&videoId={video_id}&maxResults={num_Results}'
    resp = requests.get(url)
    
    if resp.status_code == 200:
        data = resp.json()['items']
        return list(map(extract_origial_text, data))
    else: 
        return []
    

def extract_origial_text(data):
    return data['snippet']['topLevelComment']['snippet']['textOriginal']