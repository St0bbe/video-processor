import os
import uuid
import shutil
import subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, HttpUrl

from transcriber import transcrever_video
from ai_editor import analisar_video_com_gemini
from cutter import cortar_segmentos

BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "downloads"
UPLOAD_DIR = BASE_DIR / "uploads"
CLIPS_DIR = BASE_DIR / "clips"

for folder in (DOWNLOAD_DIR, UPLOAD_DIR, CLIPS_DIR):
    folder.mkdir(exist_ok=True)

app = FastAPI(title="Video Processor AI", version="2.0.0")

class VideoRequest(BaseModel):
    url: HttpUrl
    max_cortes: int = 3
    min_duracao: int = 45
    max_duracao: int = 180

@app.get("/")
def health():
    return {
        "status": "online",
        "service": "Video Processor AI",
        "version": "2.0.0"
    }

def baixar_video(url: str) -> str:
    video_id = str(uuid.uuid4())
    output_path = DOWNLOAD_DIR / f"{video_id}.mp4"

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--force-overwrites",
        "-o", str(output_path),
        url,
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail="Erro ao baixar o vídeo.") from exc

    if not output_path.exists():
        raise HTTPException(status_code=500, detail="O download terminou, mas o arquivo não foi encontrado.")

    return str(output_path)

def processar_video(video_path: str, max_cortes: int, min_duracao: int, max_duracao: int):
    if min_duracao < 15 or max_duracao <= min_duracao:
        raise HTTPException(status_code=400, detail="Intervalo de duração inválido.")

    job_id = str(uuid.uuid4())
    output_dir = CLIPS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        transcricao = transcrever_video(video_path)
        segmentos = transcricao["segments"]
        duracao = transcricao["duration"]

        melhores = analisar_video_com_gemini(
            segmentos,
            duracao,
            max_cortes=max_cortes,
            min_duracao=min_duracao,
            max_duracao=max_duracao,
        )

        if not melhores:
            raise HTTPException(status_code=422, detail="A IA não encontrou cortes válidos dentro do contexto.")

        arquivos = cortar_segmentos(video_path, str(output_dir), melhores)

        return {
            "status": "ok",
            "job_id": job_id,
            "duration": duracao,
            "language": transcricao.get("language"),
            "clips": arquivos,
            "analysis": melhores,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao processar vídeo: {exc}") from exc

@app.post("/process-url")
def process_url(data: VideoRequest):
    video_path = baixar_video(str(data.url))
    return processar_video(video_path, data.max_cortes, data.min_duracao, data.max_duracao)

@app.post("/process-upload")
async def process_upload(
    file: UploadFile = File(...),
    max_cortes: int = 3,
    min_duracao: int = 45,
    max_duracao: int = 180,
):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        raise HTTPException(status_code=400, detail="Formato de vídeo não suportado.")

    video_id = str(uuid.uuid4())
    destination = UPLOAD_DIR / f"{video_id}{suffix}"

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return processar_video(str(destination), max_cortes, min_duracao, max_duracao)
