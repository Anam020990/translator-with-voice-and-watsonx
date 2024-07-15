
from flask import Flask, request, json, render_template
from worker import speech_to_text, text_to_speech, watsonx_process_message
import base64
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_route():
    print("Processing Speech-to-Text")
    audio_binary = request.data  # Get the user's speech from their request
    text = speech_to_text(audio_binary)  # Call speech_to_text function to transcribe the speech
    # Return the response to user in JSON format
    response = app.response_class(
        response=json.dumps({'text': text}),
        status=200,
        mimetype='application/json'
    )
    print(response)
    print(response.data)
    return response

@app.route('/process-message', methods=['POST'])
def process_message_route():
    user_message = request.json['userMessage']  # Get user's message from their request
    print('user_message', user_message)
    voice = request.json['voice']  # Get user's preferred voice from their request
    print('voice', voice)
    
    # Call watsonx_process_message function to process the user's message and get a response back
    watsonx_response_text = watsonx_process_message(user_message)
    
    # Clean the response to remove any empty lines
    watsonx_response_text = os.linesep.join([s for s in watsonx_response_text.splitlines() if s])
    
    # Call our text_to_speech function to convert Watsonx API's response to speech
    watsonx_response_speech = text_to_speech(watsonx_response_text, voice)
    
    # Convert watsonx_response_speech to base64 string so it can be sent back in the JSON response
    watsonx_response_speech = base64.b64encode(watsonx_response_speech).decode('utf-8')
    
    # Send a JSON response back to the user containing their message's response both in text and speech formats
    response = app.response_class(
        response=json.dumps({
            "watsonxResponseText": watsonx_response_text,
            "watsonxResponseSpeech": watsonx_response_speech
        }),
        status=200,
        mimetype='application/json'
    )
    print(response)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
