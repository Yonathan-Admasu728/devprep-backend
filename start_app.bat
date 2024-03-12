@echo off
set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\yonin\Documents\flashcard-master-400202-25171abcd983.json
echo %GOOGLE_APPLICATION_CREDENTIALS%
uvicorn main:app --reload
