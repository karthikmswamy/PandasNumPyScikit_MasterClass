import requests

# URL of the Flask app endpoint
url = 'http://127.0.0.1:5000/similarity'

# Text data to send in the request
text_data = {
    'text': 'Blue shoe'
}

# Send the POST request
response = requests.post(url, json=text_data)

# Get the response data
response_data = response.json()

# Process the response
results = response_data['most_similar_texts']
for result in results:
    text = result['text']
    score = result['score']
    print(f'Text: {text}\nScore: {score}\n')