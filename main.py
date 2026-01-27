from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import uuid
import os

app = FastAPI()

class VideoRequest(BaseModel):
    url: str

@app.get("/")
def health():
    return {"status": "online"}

@app.post("/download")
def download_video(data: VideoRequest):
    video_id = str(uuid.uuid4())
    output_dir = "downloads"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{video_id}.mp4")

    try:
        subprocess.run(
            [
                "yt-dlp",
                "--no-playlist",
                "-f",
                "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]",
                "--merge-output-format",
                "mp4",
                "--force-overwrites",
                "--no-check-certificate",
                "--extractor-args",
                "youtube:player_client=android",
                "-o",
                output_path,
                data.url,
            ],
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail="Erro ao baixar o vídeo do YouTube"
        )

    return {
        "status": "ok",
        "file": output_path
    }
