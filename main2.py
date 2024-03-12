# import os
# import json
# import random
# import logging
# from dotenv import load_dotenv
# from datetime import datetime, timedelta
# from fastapi import FastAPI, UploadFile
# from fastapi.responses import StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# from google.cloud import texttospeech
# import openai
# import asyncio
# import hashlib
# from fastapi.responses import FileResponse
# from fastapi import BackgroundTasks
# from asyncio import sleep
# from fastapi import HTTPException
# from datetime import datetime
# import string


# from audio_utils import serve_audio
# from api_utils import transcribe_audio




# import logging
# logging.basicConfig(level=logging.DEBUG)
# logging.debug('This is a debug message')




# # Constants
# MAX_HISTORY = 100
# COMMON_RES_PATH = 'common_res.json'
# DB_PATH = 'database.json'
# LOG_PATH = 'error.log'

# # Initialize logging
# logging.basicConfig(filename=LOG_PATH, level=logging.ERROR)
# # Initialize a dictionary to store the last interaction time for each user
# last_interaction_time = {}
# # Initialize a dictionary to store the current question index for each user
# current_question_index = {}

# # Load environment variables
# load_dotenv()
# print("Environment variables loaded.")

# OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
# OPEN_AI_ORG = os.getenv("OPEN_AI_ORG")
# GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# # Initialize services
# openai.api_key = OPEN_AI_KEY
# openai.organization = OPEN_AI_ORG
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS


# # Initialize FastAPI app
# app = FastAPI()

# # CORS Configuration
# origins = ["http://localhost:5174", "http://localhost:5173", "http://localhost:8000", "http://localhost:3000"]
# app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# # Load and prepare common responses
# print("Reading from updated.json.")
# with open('updated.json', 'r') as f:
#     common_responses_list = json.load(f)
# print("Successfully read from updated.json.")


# common_responses = {
#     item['question']['text']: item['response']
#     for item in common_responses_list
# }

# # Initialize the response cache and greetings cache
# response_cache = {}
# user_greeting_cache = {}
# background_tasks = BackgroundTasks()


# # Global state dictionaries
# last_interaction_time = {}
# current_question_index = {}
# user_disengagement_status = {}
# interview_started = {}


# # FastAPI routes
# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

    

# async def check_user_response(user_token: str):
#     await asyncio.sleep(15)  # Wait for 15 seconds
#     if user_disengaged.get(user_token, False):
#         # Serve 'incorrect_elaborated_answer' and 'user_disengaged_transition' audios
#         incorrect_audio_id = common_responses_list[current_question_index[user_token]]['response']['incorrect_elaborated_answer']['audio_id']
#         incorrect_audio_file_path = f"audio_files/{incorrect_audio_id}.mp3"
#         if os.path.exists(incorrect_audio_file_path):
#             # Implement logic to serve this audio to the user
#             print(f"User {user_token} did not respond in time. Serving {incorrect_audio_file_path}")



# async def user_disengaged(user_token):
#     await sleep(15)  # Wait for 15 seconds
#     # Logic to serve 'incorrect_elaborated_answer' and 'user_disengaged_transition' audio
#     # You can adapt this part to your needs
#     print(f"User {user_token} disengaged. Serving fallback audio.")



# def is_greeting(message):
#     greetings = ["hello", "hi", "hey", "what's up", "howdy"]
#     return message.lower().strip() in greetings    




# # @app.post("/talk")

# # async def post_audio(file: UploadFile, user_token: str = None):
# #     print(f"Debug: Entered post_audio with user_token: {user_token}")
# #     if not user_token:
# #         print("Debug: User token is missing")
# #         raise HTTPException(status_code=400, detail="User token is required")
   
# #     print(f"Received POST request with user_token: {user_token}")
# #     current_time = datetime.now()

# #     print(f"Debug: last_interaction_time: {last_interaction_time}")
# #     print(f"Debug: current_question_index: {current_question_index}")

# #     if user_token not in last_interaction_time or \
# #        (current_time - last_interaction_time[user_token]).total_seconds() > 300:
# #         print("Debug: Serving welcome message")
# #         audio_file_path = "audio_files/welcome.mp3"
# #         audio_file_path = "audio_files/welcomeback.mp3" if user_token in last_interaction_time else "audio_files/welcome.mp3"

# #         last_interaction_time[user_token] = current_time
# #         if user_token not in current_question_index:
# #             current_question_index[user_token] = 0

# #         print(f"last_interaction_time: {last_interaction_time}, current_question_index: {current_question_index}")

# #         return FileResponse(audio_file_path, media_type="audio/mpeg")

# #     last_interaction_time[user_token] = current_time  # Update the last interaction time
# #     # Transcribe the user's audio
# #     user_message = transcribe_audio(file)
# #     if user_message:
# #         # Cancel the disengagement check
# #         user_disengagement_status[user_token] = False

# #         # Fetch the current question based on the index
# #         current_question_data = common_responses_list[current_question_index[user_token]]

# #         # Check if the user's response contains any of the key phrases
# #         is_correct = any(phrase.lower() in user_message.lower() for phrase in current_question_data['key_phrases'])

# #         # Choose the correct audio based on comparison
# #         if is_correct:
# #             response_data = current_question_data['response']['correct_elaborated_answer']
# #         else:
# #             response_data = current_question_data['response']['incorrect_elaborated_answer']

# #         # Serve the appropriate pre-generated audio
# #         response_audio_file_path = f"audio_files/{response_data['audio_id']}.mp3"
# #         if os.path.exists(response_audio_file_path):
# #             return FileResponse(response_audio_file_path, media_type="audio/mpeg")
        
# #          # Generate GPT-3 response as audio
# #         chat_response, response_audio_id, transition_audio_id = get_chat_response(user_message, current_question_index[user_token])
# #         audio_data, audio_file_path = await text_to_speech(chat_response)
                    
# #         if audio_data:
# #             return FileResponse(audio_file_path, media_type="audio/mpeg")
# #         else:
# #             print("Error: Failed to generate GPT-3 audio response.")
# #             raise HTTPException(status_code=500, detail="Failed to generate GPT-3 audio response.")

# #         # Increment the question index for the next question
# #     current_question_index[user_token] += 1

# #         # Schedule a background task to check for user disengagement
# #     user_disengagement_status[user_token] = True
# #     background_tasks.add_task(check_user_response, user_token)

# #         # Serve the next question
# #     next_question_audio_id = common_responses_list[current_question_index[user_token]]['question']['audio_id']
# #     next_question_audio_file_path = f"audio_files/{next_question_audio_id}.mp3"

# #     if os.path.exists(next_question_audio_file_path):
# #         return FileResponse(next_question_audio_file_path, media_type="audio/mpeg")




# def initialize_user_state(user_token, current_time):
#     if user_token not in last_interaction_time:
#         last_interaction_time[user_token] = current_time
#         current_question_index[user_token] = 0
#         user_disengagement_status[user_token] = False
#         interview_started[user_token] = False

# def is_user_disengaged(user_token, current_time):
#     return (current_time - last_interaction_time[user_token]).total_seconds() > 300

# def serve_welcome_message(user_token, current_time):
#     last_interaction_time[user_token] = current_time
#     audio_file_path = "audio_files/welcomeback.mp3"
#     return FileResponse(audio_file_path, media_type="audio/mpeg")

# def serve_greeting(user_token):
#     audio_file_path = "audio_files/welcome.mp3"
#     interview_started[user_token] = True
#     return FileResponse(audio_file_path, media_type="audio/mpeg")









# def handle_user_response(user_token, user_message, current_time, background_tasks):
#     last_interaction_time[user_token] = current_time  # Update last interaction time
    
#     # Fetch the current question based on the index
#     current_question_data = common_responses_list[current_question_index[user_token]]
    
#     # If the interview has started, check the user's answer
#     if interview_started[user_token]:
#         is_correct = any(phrase.lower() in user_message.lower() for phrase in current_question_data['key_phrases'])
        
#         # Choose the correct audio based on comparison
#         if is_correct:
#             response_data = current_question_data['response']['correct_elaborated_answer']
#         else:
#             response_data = current_question_data['response']['incorrect_elaborated_answer']
        
#         # Serve the appropriate pre-generated audio
#         response_audio_file_path = f"audio_files/{response_data['audio_id']}.mp3"
#         if os.path.exists(response_audio_file_path):
#             return FileResponse(response_audio_file_path, media_type="audio/mpeg")
#         else:
#             raise HTTPException(status_code=404, detail="Audio file not found")
    
#     # Increment the question index for the next question
#     current_question_index[user_token] += 1
    
#     # Serve the next question
#     next_question_audio_id = common_responses_list[current_question_index[user_token]]['question']['audio_id']
#     next_question_audio_file_path = f"audio_files/{next_question_audio_id}.mp3"
    
#     if os.path.exists(next_question_audio_file_path):
#         return FileResponse(next_question_audio_file_path, media_type="audio/mpeg")
#     else:
#         raise HTTPException(status_code=404, detail="Next question audio file not found")


# # @app.post("/talk")
# # async def post_audio(file: UploadFile, user_token: str = None, background_tasks: BackgroundTasks = None):
# #     print(f"Debug: Entered post_audio with user_token: {user_token}")
    
# #     if not user_token:
# #         raise HTTPException(status_code=400, detail="User token is required")
    
# #     current_time = datetime.now()
# #     initialize_user_state(user_token, current_time)
    
# #     transcription_data = transcribe_audio(file)  # Assuming transcribe_audio is a function you've defined
# #     user_message = transcription_data.get('text', '')
    
# #     if is_user_disengaged(user_token, current_time):
# #         return serve_welcome_message(user_token, current_time)
    
# #     if is_greeting(user_message):  # Assuming is_greeting is a function you've defined
# #         return serve_greeting(user_token)
    
# #     return handle_user_response(user_token, user_message, current_time, background_tasks)

# @app.post("/talk")
# async def post_audio(file: UploadFile, user_token: str = None, background_tasks: BackgroundTasks = None):
#     print(f"Debug: Entered post_audio with user_token: {user_token}")
    
#     if not user_token:
#         raise HTTPException(status_code=400, detail="User token is required")
    
#     current_time = datetime.now()
#     initialize_user_state(user_token, current_time)
    
#     # Serve the welcome message immediately upon interaction
#     welcome_audio = serve_welcome_message(user_token, current_time)
    
#     # Transcription and other logic can go here if needed
    
#     # Serve the first pre-generated interview question
#     first_question_audio = serve_first_question(user_token)
    
#     return welcome_audio or first_question_audio  # Return whichever is appropriate

    




# @app.get("/clear")
# async def clear_history():
#     file = 'database.json'
#     with open(file, 'w') as f:
#         pass
#     return {"message": "Chat history has been cleared"}


# def is_greeting(message):
#     # Remove punctuation
#     message = message.translate(str.maketrans('', '', string.punctuation))
#     greetings = ["hello", "hi", "hey", "what's up", "howdy","yes, let's go", "keep it comming", "right on!", "what are we waiting for?"]
#     return message.lower().strip() in greetings

# def serve_first_question(user_token):
#     first_question_audio_id = common_responses_list[0]['question']['audio_id']
#     first_question_audio_file_path = f"audio_files/{first_question_audio_id}.mp3"
    
#     if os.path.exists(first_question_audio_file_path):
#         return FileResponse(first_question_audio_file_path, media_type="audio/mpeg")
#     else:
#         raise HTTPException(status_code=404, detail="First question audio file not found")


# def transcribe_audio(file):
#     try:
#         with open(file.filename, 'wb') as buffer:
#             buffer.write(file.file.read())
#         audio_file = open(file.filename, "rb")

#         print("Making API call to OpenAI for transcription.")
#         transcript = openai.Audio.transcribe("whisper-1", audio_file)
#         print(f"Received transcript: {transcript}")

#         return transcript
#     except Exception as e:
#         print(f"Error in transcribing audio: {e}")
#         return None

# # New function to pre-generate audio files
# async def pre_generate_audio_files():
#     for item in common_responses_list:
#         question_text = item['question']['text']
#         question_audio_id = item['question']['audio_id']
#         response = item['response']

#         # Generate audio for question
#         await text_to_speech(question_text, question_audio_id)

#         # Generate audio for correct and incorrect elaborated answers
#         await text_to_speech(response['correct_elaborated_answer']['text'], response['correct_elaborated_answer']['audio_id'])
#         await text_to_speech(response['incorrect_elaborated_answer']['text'], response['incorrect_elaborated_answer']['audio_id'])

#         # Generate audio for transition
#         await text_to_speech(response['transition']['text'], response['transition']['audio_id'])



# def get_chat_response(user_message, current_question_idx):
#     print(f"Received user_message: {user_message}, current_question_idx: {current_question_idx}")

#     chat_response = None
#     audio_id = None
#     transition_audio_id = None

#     # Fetch the current question and answer based on the index
#     current_question_data = common_responses_list[current_question_idx]

#     # Check if the user's message matches any of the common responses
#     if user_message['text'] in common_responses:
#         response_data = common_responses[user_message['text']]
#         if is_correct_answer(user_message['text'], current_question_data['key_phrases']):
#             chat_response = response_data['correct_elaborated_answer']['text']
#             audio_id = response_data['correct_elaborated_answer']['audio_id']
#             transition_audio_id = response_data['transition']['audio_id']
#         else:
#             chat_response = response_data['incorrect_elaborated_answer']['text']
#             audio_id = response_data['incorrect_elaborated_answer']['audio_id']
#             transition_audio_id = 'tansition_disengaged.mp3'  

#     elif user_message['text'] in response_cache:
#         chat_response = response_cache[user_message['text']]
#         audio_id = hashlib.md5(chat_response.encode()).hexdigest()  # Generate a unique identifier for the chat response
#         transition_audio_id = 'tansition_disengaged.mp3'  

#     else:
#         messages = load_messages()
#         messages.append({"role": "user", "content": user_message['text']})
#         gpt_response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=messages
#         )
#         chat_response = gpt_response['choices'][0]['message']['content']
#         save_messages(user_message['text'], chat_response)
#         response_cache[user_message['text']] = chat_response  # Cache the response
#         audio_id = hashlib.md5(chat_response.encode()).hexdigest()  # Generate a unique identifier for the chat response
#         transition_audio_id = 'tansition_disengaged.mp3'  

#     return chat_response, audio_id, transition_audio_id





# def is_correct_answer(user_text, key_phrases):
#     # Implement your logic here to check if the user's answer is correct
#     # You can use the 'key_phrases' to check against the transcribed text
#     return True  # or False



# def load_messages():
#     messages = []
#     file = 'database.json'
#     empty = os.stat(file).st_size == 0
#     if not empty:
#         with open(file) as db_file:
#             data = json.load(db_file)
#             for item in data:
#                 messages.append(item)
#     else:
#         messages.append(
#             {"role": "system", "content": "You are interviewing the user for a front-end React developer position. Ask short questions that are relevant to a junior level developer. Your name is Teddy the interview master. The user is Yonathan. Keep responses under 30 words and be funny sometimes."}
#         )
#     return messages

# def save_messages(user_message, gpt_response):
#     file = 'database.json'
#     messages = load_messages()
#     messages.append({"role": "user", "content": user_message})
#     messages.append({"role": "assistant", "content": gpt_response})
#     with open(file, 'w') as f:
#         json.dump(messages, f)


# async def text_to_speech(text, audio_id=None):
#     try:
#         # Initialize the Text-to-Speech API client
#         client = texttospeech.TextToSpeechClient()

#         # Set the text input to be synthesized
#         synthesis_input = texttospeech.SynthesisInput(text=text)

#         # Build the voice request
#         voice = texttospeech.VoiceSelectionParams(
#             language_code="en-US",
#             name="en-US-Neural2-J",
#             ssml_gender=texttospeech.SsmlVoiceGender.MALE
#         )

#         # Select the type of audio file you want
#         audio_config = texttospeech.AudioConfig(
#             audio_encoding=texttospeech.AudioEncoding.LINEAR16,
#             effects_profile_id=["telephony-class-application"],
#             pitch=-4.4,
#             speaking_rate=1.1
#         )

#         # Perform the Text-to-Speech request
#         response = client.synthesize_speech(
#             input=synthesis_input, voice=voice, audio_config=audio_config
#         )

#         # Get the audio data from the response
#         audio_data = response.audio_content

#         # Generate a unique identifier for the audio file
#         if audio_id is None:
#             audio_id = hashlib.md5(text.encode()).hexdigest()

#         audio_file_path = f"{audio_id}.wav"

#         # Save the audio data to a file
#         with open(audio_file_path, "wb") as f:
#             f.write(audio_data)

#         return audio_data, audio_file_path  # Return the path along with audio data

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None

    

# def cleanup_audio(file_name):
#     try:
#         os.remove(file_name)
#         print(f"Deleted {file_name}")
#     except Exception as e:
#         print(f"Error in deleting {file_name}: {e}")



# def limit_chat_history():
#     messages = load_messages()
#     if len(messages) > MAX_HISTORY:
#         messages = messages[-MAX_HISTORY:]
#         with open('database.json', 'w') as f:
#             json.dump(messages, f)


# if __name__ == "__main__":
#     asyncio.run(pre_generate_audio_files())  # Call the function to pre-generate audio files
#     print("Google Credentials: ", GOOGLE_APPLICATION_CREDENTIALS)


# #1. Send in audio, and have it transcribed
#  #2. We want to send it to chatgpt and get a response
#  #3. We want to save the chat history to send back and forth for context.





# import secrets
# import string

# def generate_secure_token(length=32):
#     characters = string.ascii_letters + string.digits + string.punctuation
#     secure_token = ''.join(secrets.choice(characters) for i in range(length))
#     return secure_token

# # Generate and print a secure token
# token = generate_secure_token()
# print(token)
