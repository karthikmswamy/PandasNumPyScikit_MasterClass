import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Define the path to the GloVe word vectors file
glove_file = './glove.6B/glove.6B.50d.txt'

# Read the word vectors from the file
word_vectors = {}
with open(glove_file, 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype=np.float32)
        word_vectors[word] = vector

print('Pre-trained word embeddings loaded')

df = pd.read_csv('./Fashion.csv', usecols=['productDisplayName'])

# We will replace all empty data with the None string
df.productDisplayName.fillna('None', inplace=True)

# Load your dataset into a list of texts
texts = df.productDisplayName.tolist()

# Tokenize and compute word embeddings for each text in the dataset
embedding_size = 50
text_embeddings = np.zeros((len(texts), 50))
for idx, text in enumerate(texts):
    tokens = text.lower().split()
    embeddings = np.array([word_vectors[token] for token in tokens if token in word_vectors.keys()])
    if embeddings.size > 1:
        text_embedding = np.mean(embeddings, axis=0).reshape((1, embedding_size))
        text_embeddings[idx, :] = text_embedding
print('Product vectors calculated for descriptions')


# Route for the endpoint that handles the text similarity
@app.route('/similarity', methods=['POST'])
def similarity():
    # Get the input text from the request
    input_text = request.json['text']

    # Tokenize the input text
    input_tokens = input_text.lower().split()

    # Compute the average word embedding for the input text
    input_embedding = np.mean([word_vectors[token] for token in input_tokens if token in word_vectors.keys()], axis=0)

    # Compute the cosine similarity between the input embedding and all text embeddings
    similarity_scores = cosine_similarity(input_embedding.reshape(1, -1), text_embeddings)

    # Get the indices of the top three most similar texts
    top_indices = similarity_scores.argsort(axis=1)[0][-3:][::-1]

    # Get the top three most similar texts and their scores
    results = []
    for idx in top_indices:
        text = texts[idx]
        score = similarity_scores[0, idx]
        results.append({'text': text, 'score': score})

    # Return the most similar text as a response
    return jsonify({'most_similar_texts': results})

# Run the microservice
if __name__ == '__main__':

    app.run()
