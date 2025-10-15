import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import openai
import boto3
from datetime import datetime
import asyncio

# Load environment variables
load_dotenv()

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# MinIO client setup
s3_client = boto3.client(
    's3',
    endpoint_url=os.getenv("MINIO_ENDPOINT"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    region_name='us-east-1'  # MinIO ignores region
)

bucket_name = os.getenv("MINIO_BUCKET")

# FastAPI app
app = FastAPI(title="Audio Transcription API")

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Save file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Transcribe using OpenAI Whisper
    try:
        with open(temp_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file,
                response_format="text"
            )
            transcription_text = transcript
            print(transcription_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    # Save transcription as Markdown
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    md_filename = f"{timestamp}_{file.filename.rsplit('.', 1)[0]}.md"
    md_content = f"# Transcription: {file.filename}\n\n{transcription_text}"

    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=md_filename,
            Body=md_content.encode("utf-8"),
            ContentType="text/markdown"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload to MinIO: {str(e)}")

    return JSONResponse({"message": "Transcription uploaded successfully", "filename": md_filename})
