# from google.cloud import texttospeech
# from dotenv import load_dotenv
# import json
# import os

# def generate_audio(text, audio_id):
#     try:
#         client = texttospeech.TextToSpeechClient()
#         synthesis_input = texttospeech.SynthesisInput(text=text)
#         voice = texttospeech.VoiceSelectionParams(
#             language_code="en-US",
#             name="en-US-Neural2-J",
#             ssml_gender=texttospeech.SsmlVoiceGender.MALE
#         )
#         audio_config = texttospeech.AudioConfig(
#             audio_encoding=texttospeech.AudioEncoding.LINEAR16,
#             effects_profile_id=["telephony-class-application"],
#             pitch=-4.4,
#             speaking_rate=1.1  # Adjust this value to change the speed
#         )

#         response = client.synthesize_speech(
#             input=synthesis_input, voice=voice, audio_config=audio_config
#         )

#         # Ensure the directory exists
#         if not os.path.exists('audio_files'):
#             os.makedirs('audio_files')

#         with open(f"audio_files/{audio_id}.wav", "wb") as out:
#             out.write(response.audio_content)
#         print(f"Successfully generated audio for {audio_id}")

#     except Exception as e:
#         print(f"An error occurred while generating audio for {audio_id}: {e}")

# if __name__ == "__main__":
#     try:
#         with open("updated.json", "r") as f:
#             data = json.load(f)

#         for item in data:
#             question_text = item["question"]["text"]
#             question_audio_id = item["question"]["audio_id"]
#             generate_audio(question_text, question_audio_id)

#             correct_answer_text = item["response"]["correct_elaborated_answer"]["text"]
#             correct_answer_audio_id = item["response"]["correct_elaborated_answer"]["audio_id"]
#             generate_audio(correct_answer_text, correct_answer_audio_id)

#             incorrect_answer_text = item["response"]["incorrect_elaborated_answer"]["text"]
#             incorrect_answer_audio_id = item["response"]["incorrect_elaborated_answer"]["audio_id"]
#             generate_audio(incorrect_answer_text, incorrect_answer_audio_id)

#             transition_text = item["response"]["transition"]["text"]
#             transition_audio_id = item["response"]["transition"]["audio_id"]
#             generate_audio(transition_text, transition_audio_id)

#     except Exception as e:
#         print(f"An error occurred: {e}")


import requests
import json
import os
from dotenv import load_dotenv

elevenlabs_key = os.getenv("ELEVENLABS_KEY")  # Replace with your actual API key

def generate_audio(text, audio_id):
    try:
        voice_id = 'pNInz6obpgDQGcFmaJgB'
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        body = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0,
                "similarity_boost": 0,
                "style": 0.5,
                "use_speaker_boost": True
            }
        }

        headers = {
            "Content-Type": "application/json",
            "accept": "audio/mpeg",
            "xi-api-key": elevenlabs_key
        }

        response = requests.post(url, json=body, headers=headers)

        if response.status_code == 200:
            # Ensure the directory exists
            if not os.path.exists('audio_filesb'):
                os.makedirs('audio_filesb')

            with open(f"audio_filesb/{audio_id}.wav", "wb") as out:
                out.write(response.content)
            print(f"Successfully generated audio for {audio_id}")

        else:
            print(f"Failed to generate audio for {audio_id}. Status code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred while generating audio for {audio_id}: {e}")

if __name__ == "__main__":
    try:
        with open("updated2.json", "r") as f:
            data = json.load(f)

        for item in data:
            question_text = item["question"]["text"]
            question_audio_id = item["question"]["audio_id"]
            generate_audio(question_text, question_audio_id)

            correct_answer_text = item["response"]["correct_elaborated_answer"]["text"]
            correct_answer_audio_id = item["response"]["correct_elaborated_answer"]["audio_id"]
            generate_audio(correct_answer_text, correct_answer_audio_id)

            incorrect_answer_text = item["response"]["incorrect_elaborated_answer"]["text"]
            incorrect_answer_audio_id = item["response"]["incorrect_elaborated_answer"]["audio_id"]
            generate_audio(incorrect_answer_text, incorrect_answer_audio_id)

            transition_text = item["response"]["transition"]["text"]
            transition_audio_id = item["response"]["transition"]["audio_id"]
            generate_audio(transition_text, transition_audio_id)

    except Exception as e:
        print(f"An error occurred: {e}")
