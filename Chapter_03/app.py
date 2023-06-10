# 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

from flask import Flask, render_template, request, jsonify
import openai

app = Flask(__name__)

# Configure your OpenAI API credentials
openai.api_key = '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'

# Define a route to serve the index.html file
@app.route("/")
def index():
    return render_template('index.html')

# Define a route to handle the API endpoint
@app.route('/openai/completions', methods=['POST'])
def openai_completions():
    data = request.get_json()
    prompt = data['prompt']

    response = openai.Completion.create(
        engine='davinci',
        prompt=prompt,
        max_tokens=50
    )

    choices = response.choices[0].text.strip()

    return jsonify({'choices': choices})

if __name__ == '__main__':
    app.run()
