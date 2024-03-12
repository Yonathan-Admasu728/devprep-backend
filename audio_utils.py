# audio_utils.py

from fastapi.responses import FileResponse
from fastapi import HTTPException
import os

def serve_audio(audio_id: str):
    audio_file_path = f"audio_files/{audio_id}.mp3"
    if os.path.exists(audio_file_path):
        return FileResponse(audio_file_path, media_type="audio/mpeg")
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")
