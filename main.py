import os
import json
import re
from starlette.responses import JSONResponse
import random
import traceback
from fastapi import FastAPI, WebSocket
import asyncio
import time
import ast
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Body,Depends, status, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import numpy as np
from fastapi import BackgroundTasks



from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from typing import Optional

 
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.services.users import Users
from appwrite.services.account import Account




from google.cloud import texttospeech
import openai
# from openai import OpenAI

import asyncio
import hashlib
from fastapi.middleware.cors import CORSMiddleware
import string
from fastapi import File, Form
import fitz  # PyMuPDF
import shutil
import secrets
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


from audio_utils import serve_audio  # Assuming serve_audio is defined in audio_utils
from api_utils import transcribe_audio  # Assuming transcribe_audio is defined in api_utils

from pydantic import BaseModel




import logging
import uuid
from fastapi import Query
logging.basicConfig(level=logging.DEBUG)
logging.debug('This is a debug message')

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.info('Logging is configured.')


# Constants
MAX_HISTORY = 100
COMMON_RES_PATH = 'common_res.json'
DB_PATH = 'database.json'
LOG_PATH = 'error.log'

# Initialize logging
logging.basicConfig(filename=LOG_PATH, level=logging.ERROR)
# Initialize a dictionary to store the last interaction time for each user
last_interaction_time = {}
# Initialize a dictionary to store the current question index for each user
current_question_index = {}

user_states = {}

# Load environment variables
load_dotenv()
print("Environment variables loaded.")


# Initialize FastAPI app
app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPEN_AI_ORG = os.getenv("OPEN_AI_ORG")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
database_id = os.getenv('DATABASE_ID')

# Initialize services
openai.api_key = OPENAI_API_KEY
openai.organization = OPEN_AI_ORG
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS


# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# # Initialize Appwrite Client with environment variables
# client = Client()

# (client
#     .set_endpoint(os.getenv('APPWRITE_ENDPOINT')) # Your Appwrite Endpoint
#     .set_project(os.getenv('APPWRITE_PROJECT_ID')) # Your project ID from Appwrite console
#     .set_key(os.getenv('APPWRITE_API_KEY')) # Your secret API key from Appwrite console
# )

# # Initialize Appwrite Services
# users_service = Users(client)
# account_service = Account(client)

# database_service = Databases(client)



# Load environment variables
database_id = os.getenv('DATABASE_ID')
appwrite_endpoint = os.getenv('APPWRITE_ENDPOINT')
appwrite_project_id = os.getenv('APPWRITE_PROJECT_ID')
appwrite_api_key = os.getenv('APPWRITE_API_KEY')

# Initialize Appwrite Client
client = Client()
client.set_endpoint(appwrite_endpoint)
client.set_project(appwrite_project_id)
client.set_key(appwrite_api_key)

# Initialize Appwrite Services
users_service = Users(client)
account_service = Account(client)
database_service = Databases(client)

# Pydantic model for user progress
class UserProgress(BaseModel):
    userId: str
    progress: dict








app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS Configuration
origins = ["http://localhost:5174", "http://localhost:5173", "http://localhost:8000", "http://localhost:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])





class Question(BaseModel):
    id: int
    question: str
    solution: str
    starterCode: str = ""



# Load and prepare common responses
print("Reading from updated.json.")
with open('updated.json', 'r') as f:
    common_responses_list = json.load(f)
print("Successfully read from updated.json.")


common_responses = {
    item['question']['text']: item['response']
    for item in common_responses_list
}

# Load the questions from the JSON file into a list
with open('updated.json', 'r') as f:
    questions_list = json.load(f)





# Initialize the response cache and greetings cache
response_cache = {}
user_greeting_cache = {}
background_tasks = BackgroundTasks()


# Global state dictionaries
last_interaction_time = {}
current_question_index = {}
user_disengagement_status = {}
interview_started = {}
user_transcriptions = {}


# FastAPI routes
@app.get("/")
async def root():
 return {"message": "Hello World"}



# FastAPI routes
@app.post("/register")
async def register_user(email: str = Body(...), password: str = Body(...)):
    try:
        # Using Appwrite's user service to create a new user
        user = account_service.create(email=email, password=password)
        return user
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/login")
async def login_user(email: str = Body(...), password: str = Body(...)):
    try:
        # Create a session for the user
        session = account_service.create_session(email=email, password=password)
        return session
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.delete("/logout")
async def logout_user(session_id: str = Body(...)):
    try:
        # Delete the user's session
        result = account_service.delete_session(session_id=session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


# Other routes and logic...









# Helper function to get authenticated user's ID
async def get_authenticated_user_id():
    # Implement logic to retrieve the authenticated user's ID from Appwrite's account service
    # This will depend on how you've implemented user sessions in your application
    pass

# Helper function to retrieve user's progress
async def get_user_progress_data(user_id: str):
    # Implement logic to retrieve the user's progress from your database
    # This can be a query to the Appwrite database where you've stored user progress
    pass


@app.post("/store-progress")
async def store_user_progress(progress: UserProgress):
    user_id = progress.userId
    progress_data = progress.progress

    # Convert the progress data to a JSON string before storing it
    progress_json_string = json.dumps(progress_data)

    # Permissions are set dynamically for the authenticated user
    permissions = {
        'read': [f"user:{user_id}"],
        'write': [f"user:{user_id}"]
    }

    # Store the data in Appwrite
    try:
        result = database_service.create_document(
            database_id=database_id,
            collection_id='user_progress_collection',
            data={'userId': user_id, 'progress': progress_json_string},
            permissions=permissions
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))












def migrate_data(file_path, collection_id):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            # Convert to string if necessary
            item['id'] = str(item['id'])  # Ensure ID is a string if your Appwrite collection expects a string
            # Create the document in Appwrite
            database_service.create_document(
                database_id=os.getenv('DATABASE_ID'),
                collection_id=collection_id,
                data=item,
                permissions=['*']  # Set appropriate permissions
            )

def main():
    # Path to your local data directory
    data_dir = 'path/to/your/local/data/directory'

    # Migrate flashcards
    for filename in os.listdir(os.path.join(data_dir, 'flashcards')):
        if filename.endswith('Questions.json'):
            migrate_data(os.path.join(data_dir, 'flashcards', filename), '658519a0b7a765d2bb0f')

    # Migrate quizzes
    for filename in os.listdir(os.path.join(data_dir, 'quizzes')):
        if filename.endswith('Quiz.json'):
            migrate_data(os.path.join(data_dir, 'quizzes', filename), '65851b87e604041583ea')

if __name__ == '__main__':
    main()


























active_connections = {}

# Set environment variables and configuration
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'




async def websocket_heartbeat(websocket):
    try:
        while True:
            await asyncio.sleep(10)  # Send a ping every 10 seconds
            await websocket.ping()
    except Exception:
        pass  # The connection is likely closed

@app.websocket("/ws/flashcard-progress/{user_token}")
async def flashcard_progress(websocket: WebSocket, user_token: str):
    await websocket.accept()
    active_connections[user_token] = websocket
    task = asyncio.create_task(websocket_heartbeat(websocket))
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        logging.error(f"WebSocket error for user {user_token}: {e}")
    finally:
        task.cancel()
        active_connections.pop(user_token, None)


MAX_CHUNK_LENGTH = 1024  # Maximum length of text chunks for GPT processing

# Function definitions
def validate_and_sanitize_token(token):
    if not re.match(r'^[a-zA-Z0-9_\-]+$', token):
        raise ValueError("Invalid user token format.")
    return token

def extract_text_from_page(page):
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = preprocess_image_for_ocr(img)
    return pytesseract.image_to_string(img)

# def preprocess_image_for_ocr(img):
#     img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
#     _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     return Image.fromarray(img)


def preprocess_image_for_ocr(img):
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # Noise removal with median blur
    gray = cv2.medianBlur(gray, 5)
    
    # Thresholding for binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Skew correction
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Dilation to make the text more prominent
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(rotated, kernel, iterations=1)
    
    return Image.fromarray(dilated)




def extract_text_from_pdf(pdf_path):

    text_chunks = []

    def process_page(page):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = preprocess_image_for_ocr(img)
        return pytesseract.image_to_string(img)

    with fitz.open(pdf_path) as doc:
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_page = {executor.submit(process_page, page): page for page in doc}
            for future in concurrent.futures.as_completed(future_to_page):
                text_chunks.append(future.result())

    return text_chunks

def chunk_text(text, max_length):
    chunks = []
    current_chunk = ""
    for sentence in text.split('.'):
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + '.'
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + '.'
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

async def generate_flashcards_from_text(text_chunks):
    all_flashcards = []
    for chunk in text_chunks:
        chunk_flashcards = await process_chunk_for_flashcards(chunk)
        all_flashcards.extend(chunk_flashcards)
    return all_flashcards

# async def process_chunk_for_flashcards(chunk):
#     if chunk.strip() == "":
#         return []
#     logging.info(f"Generating flashcards from chunk: {chunk[:50]}...")
#     prompt = f"Create flashcards in the following format:\nFlashcard 1: Front: [Question] / Back: [Answer]\n...\nBased on this text: {chunk}"
#     gpt_response = get_gpt3_response(prompt, max_tokens=4096)
#     return parse_gpt_response_to_flashcards(gpt_response) if gpt_response else []

async def process_chunk_for_flashcards(chunk):
    if chunk.strip() == "":
        return []

    logging.info(f"Generating flashcards from chunk: {chunk[:50]}...")

    # Refined prompt to encourage simple Q&A pairs
    prompt = (
        "Based on the following text, generate a series of flashcards as simple question and answer pairs. "
        "Start each flashcard with a question, followed by its answer, without using labels like 'Question' or 'Answer'.\n\n"
        f"Text: {chunk}\n\n"
        "Example:\n"
        "What is React?\n"
        "React is a JavaScript library for building user interfaces, primarily for single-page applications. "
        "It's used for handling the view layer.\n"
        "What are React components?\n"
        "Components are the building blocks of a React application's UI. "
        "They split the UI into reusable, independent pieces."
    )

    gpt_response = get_gpt3_response(prompt, max_tokens=4096)

    # Ensure 'parse_gpt_response_to_flashcards' can handle this expected Q&A format
    return parse_gpt_response_to_flashcards(gpt_response) if gpt_response else []


def get_gpt3_response(prompt, max_tokens=60000):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=max_tokens
        )
        logging.debug(f"GPT-3 Response: {response.choices[0].message.content.strip()}")
        return response.choices[0].message.content.strip() if response.choices else ""
    except Exception as e:
        logging.error(f"Error calling GPT-3 API: {e}")
        return None


def parse_gpt_response_to_flashcards(gpt_response):
    flashcards = []
    lines = gpt_response.split('\n')

    # Assuming every odd line is a question and every even line is an answer
    for i in range(0, len(lines), 2):
        if i+1 < len(lines):  # Ensure there's a pair
            question = lines[i].strip()
            answer = lines[i+1].strip()

            # Add the Q&A pair if both question and answer are present
            if question and answer:
                flashcards.append({"question": question, "answer": answer})

    if not flashcards:
        logging.warning("No flashcards found in GPT-3 response.")
    else:
        logging.info(f"Extracted {len(flashcards)} flashcards from GPT-3 response.")

    return flashcards






# def store_flashcards(flashcards, upload_name):
#     if not flashcards:
#         logging.warning("No flashcards to store.")
#         return
#     directory = "user_data"
#     os.makedirs(directory, exist_ok=True)
#     file_path = os.path.join(directory, f"{upload_name}_flashcards.json")
#     with open(file_path, "w") as file:
#         json.dump(flashcards, file)
#     logging.info(f"Flashcards stored successfully in {file_path}.")


def store_flashcards(flashcards, upload_name):
    if not flashcards:
        logging.warning("No flashcards to store.")
        return

    # Define the directory where the flashcards will be stored
    directory = "user_data"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

    # Define the full path for the JSON file
    file_path = os.path.join(directory, f"{upload_name}_flashcards.json")

    # Write the flashcards to the JSON file with proper formatting
    try:
        with open(file_path, "w") as file:
            json.dump(flashcards, file, indent=4)  # Use indent=4 for pretty printing
        logging.info(f"Flashcards stored successfully in {file_path}.")
    except Exception as e:
        logging.error(f"Failed to store flashcards: {e}")




async def perform_ocr(file_path: str, user_token: str):
    try:
         # Reset or reinitialize state for the user token at the start
        reset_user_state(user_token)

        websocket = active_connections.get(user_token)
        if websocket:
            await websocket.send_text("OCR processing started")

        # Text extraction from PDF
        text_chunks = extract_text_from_pdf(file_path)

        if websocket:
            await websocket.send_text("Text extraction complete")

        # Await the asynchronous flashcard generation
        flashcards = await generate_flashcards_from_text(text_chunks)

        if websocket:
            await websocket.send_text("Flashcard generation complete")

        # Store the flashcards
        store_flashcards(flashcards, secure_filename(file_path))

    except Exception as e:
        logging.error(f"Error during OCR processing for {file_path}: {e}")
        if user_token in active_connections:
            await active_connections[user_token].send_text(f"Error during OCR processing: {str(e)}")
    finally:
        # Clear the state for the user token on completion or error
        clear_user_state(user_token)

def reset_user_state(user_token):
    # Assuming 'user_states' is a dictionary managing states for different users
    if user_token in user_states:
        user_states[user_token] = {'ocr_started': False, 'text_extracted': False, 'flashcards_generated': False, 'error': None}
    else:
        user_states[user_token] = {'ocr_started': True, 'text_extracted': False, 'flashcards_generated': False, 'error': None}


def clear_user_state(user_token):
    # Remove the user token from the 'user_states' dictionary to clear the state
    user_states.pop(user_token, None)




def secure_filename(filename):
    """
    Sanitize the filename to avoid directory traversal or insecure file names.
    """
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    return filename



@app.post("/upload-pdf")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...), user_token: str = Form(...)):
    try:
        # Assume validate_and_sanitize_token and other necessary functions are defined elsewhere
        user_token = validate_and_sanitize_token(user_token)
        upload_dir = 'uploads'

        # Use the secure_filename function to sanitize the uploaded file's name
        secure_file_name = secure_filename(file.filename)
        file_location = os.path.join(upload_dir, secure_file_name)

        if not file.content_type == 'application/pdf':
            raise HTTPException(status_code=400, detail="Invalid file type")

        os.makedirs(upload_dir, exist_ok=True)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"File {secure_file_name} uploaded successfully")

        # Offload OCR processing to a background task
        background_tasks.add_task(perform_ocr, file_location, user_token)

        return {"message": "File upload successful, processing started."}

    except Exception as e:
        logging.error(f"Error during file upload: {e}")
        if user_token in active_connections:
            await active_connections[user_token].send_text(f"Error during file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during file upload: {str(e)}")

    

@app.get("/flashcards/custom/{upload_name}")
async def get_custom_flashcards(upload_name: str):
    directory = "user_data"
    file_name = f"{upload_name}_flashcards.json"
    file_path = os.path.join(directory, file_name)

    # Sanitize upload_name to prevent directory traversal
    if ".." in upload_name or "/" in upload_name:
        raise HTTPException(status_code=400, detail="Invalid upload name")

    if not os.path.exists(file_path):
        logging.error(f"Requested flashcards file not found: {file_path}")
        raise HTTPException(status_code=404, detail="Flashcards not found")

    # Use FileResponse for efficient file serving
    return FileResponse(file_path)


























# Function to load JSON data
def load_json_data(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Global variable to store algorithm questions
algo_questions = load_json_data('data/flashcards/algoQuestions.json')

@app.get("/flashcards/menu")
async def get_flashcard_menu():
    file_path = 'data/flashCardsData.json'
    try:
        return load_json_data(file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Flashcard menu data not found.")

@app.get("/flashcards/{framework}")
async def get_flashcards(framework: str):
    if "algo" in framework:  # Check if the request is for algorithmic flashcards
        # Filter and return algorithmic questions based on the language (included in the framework key)
        language = "python" if "python" in framework else "javascript"
        filtered_questions = [{
            "id": q["id"],
            "title": q["title"],  # Include the title in the response
            "question": q["question"],
            "solution": q["solutions"][language]  # Return the solution in the requested language
        } for q in algo_questions]
        return filtered_questions
    else:
        # Handle non-algorithmic frameworks
        file_path = f'data/flashcards/{framework}Questions.json'
        try:
            return load_json_data(file_path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Flashcards for {framework} not found.")


@app.get("/quizzes/{framework}")
async def get_quizzes(framework: str):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Check if the requested framework is algorithm-related
    if "algo" in framework:
        # Path to the shared algo quiz file
        file_path = os.path.join(dir_path, 'data/quizzes/algo-quiz.json')
    else:
        # Path to the framework-specific quiz file
        file_path = os.path.join(dir_path, f'data/quizzes/{framework}Quiz.json')

    try:
        return load_json_data(file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Quizzes for {framework} not found.")
    


@app.get("/questions")
async def get_questions():
    return JSONResponse(content=algo_questions)

@app.get("/questions/{question_id}")
async def get_question(question_id: int, language: str = Query(None, enum=["python", "javascript"])):
    question = next((q for q in algo_questions if q["id"] == question_id), None)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    # Return all details but highlight starter code for the specified language if provided
    question_data = question.copy()  # Assuming this is a dictionary
    if language:
        question_data["starterCode"] = question_data.get("starterCode", {}).get(language, "")
    return JSONResponse(content=question_data)


@app.get("/solutions/{question_id}")
async def get_solution(question_id: int):
    question = next((q for q in algo_questions if q["id"] == question_id), None)
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found")
    return JSONResponse(content=question)














async def check_user_response(user_token: str):
    await asyncio.sleep(15)  # Wait for 15 seconds
    if user_disengagement_status.get(user_token, False):
        incorrect_audio_id = common_responses_list[current_question_index[user_token]]['response']['incorrect_elaborated_answer']['audio_id']
        serve_audio(incorrect_audio_id)  # Using the utility function here

@app.post("/talk")
async def post_audio(file: UploadFile, user_token: str = None, background_tasks: BackgroundTasks = None):
    print(f"Received user_token: {user_token}")
    print(f"Debug: Entered post_audio with user_token: {user_token}")
    
    if not user_token:
        raise HTTPException(status_code=400, detail="User token is required")
    
    current_time = datetime.now()
    initialize_user_state(user_token, current_time)
    
        # Transcribe the user's voice to text
    transcription_data = transcribe_audio(file)

    if transcription_data is None or not isinstance(transcription_data, dict):
        print("Error in transcribing audio or invalid data format.")
        # Handle the error appropriately, perhaps by returning a response or raising an exception
    else:
        user_message = transcription_data.get('text', '').strip()
        


    # Save the transcription
    user_transcriptions[user_token] = user_message

    # Check for user disengagement
    if is_user_disengaged(user_token, current_time):
        return serve_welcome_message(user_token, current_time)
    
    # Check if the message is a greeting
    if is_greeting(user_message):
        return serve_greeting(user_token)
    
    # Fetch the current question based on the index
    current_question_data = common_responses_list[current_question_index[user_token]]
    
    # Check if the user's answer contains any of the key phrases
    if any(phrase.lower() in user_message.lower() for phrase in current_question_data['key_phrases']):
        # Serve the correct response
        correct_audio_id = current_question_data['response']['correct_elaborated_answer']['audio_id']
        correct_audio_file_path = f"audio_files/{correct_audio_id}.mp3"
        if os.path.exists(correct_audio_file_path):
            return FileResponse(correct_audio_file_path, media_type="audio/mpeg")
    else:
        # Fallback to GPT-3 for unrelated questions or jokes
        gpt3_response = get_gpt3_response(user_message)
        
        # Convert GPT-3 response to audio
        audio_data, audio_file_path = await text_to_speech(gpt3_response)
        
        # Serve the audio file
        return FileResponse(audio_file_path, media_type="audio/wav")
    
    # Increment the question index for the next question
    current_question_index[user_token] += 1
    
    # Serve the next question
    return serve_question(user_token)



@app.get("/get_transcription")
async def get_transcription(user_token: str):
    if not user_token:
        return {"transcription": "User token is null"}
    if user_token in user_transcriptions:
        return {"transcription": user_transcriptions[user_token]}
    else:
        return {"transcription": "No transcription available."}

    

@app.get("/get_question/{index}")
async def get_question(index: int):
    print(f"Received request for question {index}")
    try:
        question_data = questions_list[index]
        return {"question": question_data["question"]["text"]}
    except IndexError:
        return {"error": "Question index out of range"}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/clear")
async def clear_history():
    file = 'database.json'
    with open(file, 'w') as f:
        pass
    return {"message": "Chat history has been cleared"}


# New function to serve a question based on the user's token and question index
def serve_question(user_token):
    question_audio_id = common_responses_list[current_question_index[user_token]]['question']['audio_id']
    question_audio_file_path = f"audio_files/{question_audio_id}.mp3"
    
    if os.path.exists(question_audio_file_path):
        return FileResponse(question_audio_file_path, media_type="audio/mpeg")
    else:
        raise HTTPException(status_code=404, detail="Question audio file not found")
    


# def get_gpt3_response(prompt):
#     openai.api_key = "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"  # Make sure to set your actual API key
#     model_engine = "text-davinci-002"  # You can use other engines like "text-ada" based on your needs
#     response = openai.Completion.create(
#         engine=model_engine,
#         prompt=prompt,
#         max_tokens=100  # Limit the response to 100 tokens
#     )
#     return response.choices[0].text.strip()





def initialize_user_state(user_token, current_time):
    if user_token not in last_interaction_time:
        last_interaction_time[user_token] = current_time
        current_question_index[user_token] = 0
        user_disengagement_status[user_token] = False
        interview_started[user_token] = False

def is_user_disengaged(user_token, current_time):
    return (current_time - last_interaction_time[user_token]).total_seconds() > 300

def serve_welcome_message(user_token, current_time):
    last_interaction_time[user_token] = current_time
    audio_file_path = "audio_files/welcomeback.mp3"
    return FileResponse(audio_file_path, media_type="audio/mpeg")

def serve_greeting(user_token):
    # Serve a greeting message or audio file
    return {"message": "Hello, welcome back!"}



def handle_user_response(user_token, user_message, current_time, background_tasks):
    last_interaction_time[user_token] = current_time  # Update last interaction time
    
    # Fetch the current question based on the index
    current_question_data = common_responses_list[current_question_index[user_token]]
    
    # If the interview has started, check the user's answer
    if interview_started[user_token]:
        is_correct = any(phrase.lower() in user_message.lower() for phrase in current_question_data['key_phrases'])
        
        # Choose the correct audio based on comparison
        if is_correct:
            response_data = current_question_data['response']['correct_elaborated_answer']
        else:
            response_data = current_question_data['response']['incorrect_elaborated_answer']
        
        # Serve the appropriate pre-generated audio
        response_audio_file_path = f"audio_files/{response_data['audio_id']}.mp3"
        if os.path.exists(response_audio_file_path):
            return FileResponse(response_audio_file_path, media_type="audio/mpeg")
        else:
            raise HTTPException(status_code=404, detail="Audio file not found")
    
    # Increment the question index for the next question
    current_question_index[user_token] += 1
    
    # Serve the next question
    next_question_audio_id = common_responses_list[current_question_index[user_token]]['question']['audio_id']
    next_question_audio_file_path = f"audio_files/{next_question_audio_id}.mp3"
    
    if os.path.exists(next_question_audio_file_path):
        return FileResponse(next_question_audio_file_path, media_type="audio/mpeg")
    else:
        raise HTTPException(status_code=404, detail="Next question audio file not found")


def is_greeting(message):
    # Remove punctuation
    message = message.translate(str.maketrans('', '', string.punctuation))
    greetings = ["hello", "hi", "hey", "what's up", "howdy","yes, let's go", "keep it comming", "right on!", "what are we waiting for?"]
    return message.lower().strip() in greetings

def serve_first_question(user_token):
    first_question_audio_id = common_responses_list[0]['question']['audio_id']
    first_question_audio_file_path = f"audio_files/{first_question_audio_id}.mp3"
    
    if os.path.exists(first_question_audio_file_path):
        return FileResponse(first_question_audio_file_path, media_type="audio/mpeg")
    else:
        raise HTTPException(status_code=404, detail="First question audio file not found")


def transcribe_audio(file):
    try:
        with open(file.filename, 'wb') as buffer:
            buffer.write(file.file.read())
        audio_file = open(file.filename, "rb")

        print("Making API call to OpenAI for transcription.")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        print(f"Received transcript: {transcript}")

        return transcript
    except Exception as e:
        print(f"Error in transcribing audio: {e}")
        return None

# New function to pre-generate audio files
async def pre_generate_audio_files():
    for item in common_responses_list:
        question_text = item['question']['text']
        question_audio_id = item['question']['audio_id']
        response = item['response']

        # Generate audio for question
        await text_to_speech(question_text, question_audio_id)

        # Generate audio for correct and incorrect elaborated answers
        await text_to_speech(response['correct_elaborated_answer']['text'], response['correct_elaborated_answer']['audio_id'])
        await text_to_speech(response['incorrect_elaborated_answer']['text'], response['incorrect_elaborated_answer']['audio_id'])

        # Generate audio for transition
        await text_to_speech(response['transition']['text'], response['transition']['audio_id'])



def get_chat_response(user_message, current_question_idx):
    print(f"Received user_message: {user_message}, current_question_idx: {current_question_idx}")

    chat_response = None
    audio_id = None
    transition_audio_id = None

    # Fetch the current question and answer based on the index
    current_question_data = common_responses_list[current_question_idx]

    # Check if the user's message matches any of the common responses
    if user_message['text'] in common_responses:
        response_data = common_responses[user_message['text']]
        if is_correct_answer(user_message['text'], current_question_data['key_phrases']):
            chat_response = response_data['correct_elaborated_answer']['text']
            audio_id = response_data['correct_elaborated_answer']['audio_id']
            transition_audio_id = response_data['transition']['audio_id']
        else:
            chat_response = response_data['incorrect_elaborated_answer']['text']
            audio_id = response_data['incorrect_elaborated_answer']['audio_id']
            transition_audio_id = 'tansition_disengaged.mp3'  

    elif user_message['text'] in response_cache:
        chat_response = response_cache[user_message['text']]
        audio_id = hashlib.md5(chat_response.encode()).hexdigest()  # Generate a unique identifier for the chat response
        transition_audio_id = 'tansition_disengaged.mp3'  

    else:
        messages = load_messages()
        messages.append({"role": "user", "content": user_message['text']})
        gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        chat_response = gpt_response['choices'][0]['message']['content']
        save_messages(user_message['text'], chat_response)
        response_cache[user_message['text']] = chat_response  # Cache the response
        audio_id = hashlib.md5(chat_response.encode()).hexdigest()  # Generate a unique identifier for the chat response
        transition_audio_id = 'tansition_disengaged.mp3'  

    return chat_response, audio_id, transition_audio_id





def is_correct_answer(user_text, key_phrases):
    # Implement your logic here to check if the user's answer is correct
    # You can use the 'key_phrases' to check against the transcribed text
    return True  # or False



def load_messages():
    messages = []
    file = 'database.json'
    empty = os.stat(file).st_size == 0
    if not empty:
        with open(file) as db_file:
            data = json.load(db_file)
            for item in data:
                messages.append(item)
    else:
        messages.append(
            {"role": "system", "content": "You are interviewing the user for a front-end React developer position. Ask short questions that are relevant to a junior level developer. Your name is Teddy the interview master. The user is Yonathan. Keep responses under 30 words and be funny sometimes."}
        )
    return messages

def save_messages(user_message, gpt_response):
    file = 'database.json'
    messages = load_messages()
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": gpt_response})
    with open(file, 'w') as f:
        json.dump(messages, f)


async def text_to_speech(text, audio_id=None):
    try:
        # Initialize the Text-to-Speech API client
        client = texttospeech.TextToSpeechClient()

        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-J",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )

        # Select the type of audio file you want
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            effects_profile_id=["telephony-class-application"],
            pitch=-4.4,
            speaking_rate=1.1
        )

        # Perform the Text-to-Speech request
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Get the audio data from the response
        audio_data = response.audio_content

        # Generate a unique identifier for the audio file
        if audio_id is None:
            audio_id = hashlib.md5(text.encode()).hexdigest()

        audio_file_path = f"{audio_id}.wav"

        # Save the audio data to a file
        with open(audio_file_path, "wb") as f:
            f.write(audio_data)

        return audio_data, audio_file_path  # Return the path along with audio data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    

def cleanup_audio(file_name):
    try:
        os.remove(file_name)
        print(f"Deleted {file_name}")
    except Exception as e:
        print(f"Error in deleting {file_name}: {e}")



def limit_chat_history():
    messages = load_messages()
    if len(messages) > MAX_HISTORY:
        messages = messages[-MAX_HISTORY:]
        with open('database.json', 'w') as f:
            json.dump(messages, f)


if __name__ == "__main__":
    asyncio.run(pre_generate_audio_files())  # Call the function to pre-generate audio files
    print("Google Credentials: ", GOOGLE_APPLICATION_CREDENTIALS)



if __name__ == "__main__":
    # Start the FastAPI application
    # Use Uvicorn or a similar ASGI server to run your app
    pass


#1. Send in audio, and have it transcribed
 #2. We want to send it to chatgpt and get a response
 #3. We want to save the chat history to send back and forth for context.




