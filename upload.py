

# active_connections = {}

# # Set environment variables and configuration
# os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'




# async def websocket_heartbeat(websocket):
#     try:
#         while True:
#             await asyncio.sleep(10)  # Send a ping every 10 seconds
#             await websocket.ping()
#     except Exception:
#         pass  # The connection is likely closed

# @app.websocket("/ws/flashcard-progress/{user_token}")
# async def flashcard_progress(websocket: WebSocket, user_token: str):
#     await websocket.accept()
#     active_connections[user_token] = websocket
#     task = asyncio.create_task(websocket_heartbeat(websocket))
#     try:
#         while True:
#             await websocket.receive_text()
#     except Exception as e:
#         logging.error(f"WebSocket error for user {user_token}: {e}")
#     finally:
#         task.cancel()
#         active_connections.pop(user_token, None)


# MAX_CHUNK_LENGTH = 1024  # Maximum length of text chunks for GPT processing

# # Function definitions
# def validate_and_sanitize_token(token):
#     if not re.match(r'^[a-zA-Z0-9_\-]+$', token):
#         raise ValueError("Invalid user token format.")
#     return token

# def extract_text_from_page(page):
#     pix = page.get_pixmap()
#     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#     img = preprocess_image_for_ocr(img)
#     return pytesseract.image_to_string(img)

# def preprocess_image_for_ocr(img):
#     img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
#     _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     return Image.fromarray(img)



# def extract_text_from_pdf(pdf_path):

#     text_chunks = []

#     def process_page(page):
#         pix = page.get_pixmap()
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         img = preprocess_image_for_ocr(img)
#         return pytesseract.image_to_string(img)

#     with fitz.open(pdf_path) as doc:
#         with ThreadPoolExecutor(max_workers=5) as executor:
#             future_to_page = {executor.submit(process_page, page): page for page in doc}
#             for future in concurrent.futures.as_completed(future_to_page):
#                 text_chunks.append(future.result())

#     return text_chunks

# def chunk_text(text, max_length):
#     chunks = []
#     current_chunk = ""
#     for sentence in text.split('.'):
#         if len(current_chunk) + len(sentence) + 1 <= max_length:
#             current_chunk += sentence + '.'
#         else:
#             chunks.append(current_chunk)
#             current_chunk = sentence + '.'
#     if current_chunk:
#         chunks.append(current_chunk)
#     return chunks

# async def generate_flashcards_from_text(text_chunks):
#     all_flashcards = []
#     for chunk in text_chunks:
#         chunk_flashcards = await process_chunk_for_flashcards(chunk)
#         all_flashcards.extend(chunk_flashcards)
#     return all_flashcards

# async def process_chunk_for_flashcards(chunk):
#     if chunk.strip() == "":
#         return []
#     logging.info(f"Generating flashcards from chunk: {chunk[:50]}...")
#     prompt = f"Create flashcards in the following format:\nFlashcard 1: Front: [Question] / Back: [Answer]\n...\nBased on this text: {chunk}"
#     gpt_response = get_gpt3_response(prompt, max_tokens=4096)
#     return parse_gpt_response_to_flashcards(gpt_response) if gpt_response else []

# def get_gpt3_response(prompt, max_tokens=60000):
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4-1106-preview",
#             messages=[{"role": "system", "content": prompt}],
#             max_tokens=max_tokens
#         )
#         return response.choices[0].message.content.strip() if response.choices else ""
#     except Exception as e:
#         logging.error(f"Error calling GPT-3 API: {e}")
#         return None


# def parse_gpt_response_to_flashcards(gpt_response):
#     flashcards = []
#     for line in gpt_response.split('\n'):
#         if line.startswith("Flashcard"):
#             parts = line.split("Front:", 1)
#             if len(parts) > 1:
#                 front_back = parts[1].split("/ Back:")
#                 if len(front_back) == 2:
#                     flashcards.append({"question": front_back[0].strip(), "answer": front_back[1].strip()})
#     if not flashcards:
#         logging.warning(f"No flashcards found in GPT-3 response: {gpt_response}")
#     return flashcards


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




# async def perform_ocr(file_path: str, user_token: str):
#     try:
#          # Reset or reinitialize state for the user token at the start
#         reset_user_state(user_token)

#         websocket = active_connections.get(user_token)
#         if websocket:
#             await websocket.send_text("OCR processing started")

#         # Text extraction from PDF
#         text_chunks = extract_text_from_pdf(file_path)

#         if websocket:
#             await websocket.send_text("Text extraction complete")

#         # Await the asynchronous flashcard generation
#         flashcards = await generate_flashcards_from_text(text_chunks)

#         if websocket:
#             await websocket.send_text("Flashcard generation complete")

#         # Store the flashcards
#         store_flashcards(flashcards, secure_filename(file_path))

#     except Exception as e:
#         logging.error(f"Error during OCR processing for {file_path}: {e}")
#         if user_token in active_connections:
#             await active_connections[user_token].send_text(f"Error during OCR processing: {str(e)}")
#     finally:
#         # Clear the state for the user token on completion or error
#         clear_user_state(user_token)

# def reset_user_state(user_token):
#     # Assuming 'user_states' is a dictionary managing states for different users
#     if user_token in user_states:
#         user_states[user_token] = {'ocr_started': False, 'text_extracted': False, 'flashcards_generated': False, 'error': None}
#     else:
#         user_states[user_token] = {'ocr_started': True, 'text_extracted': False, 'flashcards_generated': False, 'error': None}


# def clear_user_state(user_token):
#     # Remove the user token from the 'user_states' dictionary to clear the state
#     user_states.pop(user_token, None)




# def secure_filename(filename):
#     """
#     Sanitize the filename to avoid directory traversal or insecure file names.
#     """
#     filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
#     return filename



# @app.post("/upload-pdf")
# async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...), user_token: str = Form(...)):
#     try:
#         # Assume validate_and_sanitize_token and other necessary functions are defined elsewhere
#         user_token = validate_and_sanitize_token(user_token)
#         upload_dir = 'uploads'

#         # Use the secure_filename function to sanitize the uploaded file's name
#         secure_file_name = secure_filename(file.filename)
#         file_location = os.path.join(upload_dir, secure_file_name)

#         if not file.content_type == 'application/pdf':
#             raise HTTPException(status_code=400, detail="Invalid file type")

#         os.makedirs(upload_dir, exist_ok=True)
#         with open(file_location, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         logging.info(f"File {secure_file_name} uploaded successfully")

#         # Offload OCR processing to a background task
#         background_tasks.add_task(perform_ocr, file_location, user_token)

#         return {"message": "File upload successful, processing started."}

#     except Exception as e:
#         logging.error(f"Error during file upload: {e}")
#         if user_token in active_connections:
#             await active_connections[user_token].send_text(f"Error during file upload: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error during file upload: {str(e)}")

    

# @app.get("/flashcards/custom/{upload_name}")
# async def get_custom_flashcards(upload_name: str):
#     directory = "user_data"
#     file_name = f"{upload_name}_flashcards.json"
#     file_path = os.path.join(directory, file_name)

#     # Sanitize upload_name to prevent directory traversal
#     if ".." in upload_name or "/" in upload_name:
#         raise HTTPException(status_code=400, detail="Invalid upload name")

#     if not os.path.exists(file_path):
#         logging.error(f"Requested flashcards file not found: {file_path}")
#         raise HTTPException(status_code=404, detail="Flashcards not found")

#     # Use FileResponse for efficient file serving
#     return FileResponse(file_path)
