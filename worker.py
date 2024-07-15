import requests
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

# Watson Machine Learning setup
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    #"apikey": 'API_KEY'
}
project_id = "skills-network"
model_id = ModelTypes.FLAN_UL2
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024
}
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

# Watson Speech-to-Text setup
def speech_to_text(audio_binary):
    base_url = '...'  # Replace with your actual base URL
    api_url = f'{base_url}/speech-to-text/api/v1/recognize'
    params = {
        'model': 'en-US_Multimedia',
    }
    response = requests.post(api_url, params=params, data=audio_binary)
    if response.status_code == 200:
        response_json = response.json()
        print('Speech-to-Text response:', response_json)
        if 'results' in response_json and response_json['results']:
            transcript = response_json['results'][0]['alternatives'][0]['transcript']
            print('Recognized text:', transcript)
            return transcript
        else:
            print('No results found in the response.')
            return None
    else:
        print(f'Error: HTTP status code {response.status_code}')
        return None

# Watson Text-to-Speech setup
def text_to_speech(text, voice=""):
    base_url = '...'  # Replace with your actual base URL
    api_url = f'{base_url}/text-to-speech/api/v1/synthesize'
    api_url += '?output=audio/wav'
    if voice and voice != "default":
        api_url += f'&voice={voice}'
    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }
    json_data = {
        'text': text,
    }
    response = requests.post(api_url, headers=headers, json=json_data)
    if response.status_code == 200:
        print('Text-to-Speech response:', response)
        return response.content
    else:
        print(f'Error: HTTP status code {response.status_code}')
        return None

# Watsonx function using Watson Machine Learning
def watsonx_process_message(user_message):
    prompt = f"""You are an assistant helping translate sentences from English into Spanish.
    Translate the query to Spanish: ```{user_message}```."""
    response_text = model.generate_text(prompt=prompt)
    print("watsonx response:", response_text)
    return response_text
