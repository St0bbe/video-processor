import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, HttpUrl

from transcriber import transcrever_video
from ai_editor import analisar_video_com_gemini
from cutter import cortar_segmentos

BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "downloads"
UPLOAD_DIR = BASE_DIR / "uploads"
CLIPS_DIR = BASE_DIR / "clips"
JOBS_DIR = BASE_DIR / "jobs"

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "4096"))
MAX_DOWNLOADED_VIDEO_MB = int(os.getenv("MAX_DOWNLOADED_VIDEO_MB", "8192"))
MAX_VIDEO_MINUTES = int(os.getenv("MAX_VIDEO_MINUTES", "240"))
RETENTION_HOURS = int(os.getenv("RETENTION_HOURS", "48"))

for folder in (DOWNLOAD_DIR, UPLOAD_DIR, CLIPS_DIR, JOBS_DIR):
    folder.mkdir(exist_ok=True)

app = FastAPI(
    title="Cortes Inteligentes com IA",
    summary="Transforme vídeos longos em cortes dos melhores momentos.",
    description="""
## 🎬 Gerador de cortes inteligentes

Envie um **link de vídeo** ou faça **upload de um arquivo**. O sistema:

1. baixa/recebe o vídeo;
2. transcreve o áudio com Whisper;
3. analisa o contexto com Gemini;
4. escolhe os melhores momentos;
5. gera os cortes automaticamente com FFmpeg.

### Como usar com um link
Abra **🎬 Processar vídeo → Analisar vídeo por link**, clique em **Try it out**, cole a URL e clique em **Execute**.

Depois copie o `job_id` retornado e consulte **📊 Acompanhar processamento**.

Quando o status for `done`, baixe os cortes em **⬇️ Baixar resultados**.
""",
    version="3.4.1",
    contact={"name": "Video Processor AI"},
    openapi_tags=[
        {"name": "🎬 Processar vídeo", "description": "Envie um link ou arquivo para encontrar e gerar os melhores cortes."},
        {"name": "📊 Acompanhar processamento", "description": "Veja o progresso da análise e o resultado do processamento."},
        {"name": "⬇️ Baixar resultados", "description": "Baixe individualmente os cortes gerados."},
        {"name": "⚙️ Sistema", "description": "Status e manutenção da aplicação."},
    ],
)

class VideoRequest(BaseModel):
    url: HttpUrl
    max_cortes: int = Field(default=3, ge=1, le=10)
    min_duracao: int = Field(default=45, ge=15, le=600)
    max_duracao: int = Field(default=180, ge=20, le=900)

def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"

def _save_job(job_id: str, data: dict):
    path = _job_path(job_id)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)

def _load_job(job_id: str) -> dict:
    path = _job_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job não encontrado.")
    return json.loads(path.read_text(encoding="utf-8"))

def _update_job(job_id: str, **updates):
    job = _load_job(job_id)
    job.update(updates)
    job["updated_at"] = time.time()
    _save_job(job_id, job)
    return job

def _validate_durations(min_duracao: int, max_duracao: int):
    if max_duracao <= min_duracao:
        raise HTTPException(status_code=400, detail="max_duracao deve ser maior que min_duracao.")

def _probe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Não foi possível obter a duração do vídeo.")
    return float(result.stdout.strip())

def _validate_video_limits(video_path: str, source_type: str = "upload"):
    size_mb = Path(video_path).stat().st_size / (1024 * 1024)
    size_limit = MAX_DOWNLOADED_VIDEO_MB if source_type == "url" else MAX_UPLOAD_MB
    if size_mb > size_limit:
        raise ValueError(f"Arquivo excede o limite de {size_limit} MB.")

    duration = _probe_duration(video_path)
    if duration > MAX_VIDEO_MINUTES * 60:
        raise ValueError(f"Vídeo excede o limite de {MAX_VIDEO_MINUTES} minutos.")

def _cleanup_old_files():
    cutoff = time.time() - (RETENTION_HOURS * 3600)
    for directory in (DOWNLOAD_DIR, UPLOAD_DIR, CLIPS_DIR, JOBS_DIR):
        for item in directory.iterdir():
            try:
                if item.stat().st_mtime >= cutoff:
                    continue
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except Exception:
                pass

@app.on_event("startup")
def startup_cleanup():
    _cleanup_old_files()

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def interface():
    page = BASE_DIR / "static" / "index.html"
    if not page.exists():
        return HTMLResponse(
            "<h1>Interface não encontrada.</h1><p>Verifique static/index.html.</p>",
            status_code=500,
        )
    return HTMLResponse(page.read_text(encoding="utf-8"))

@app.get("/api/status", tags=["⚙️ Sistema"], summary="Verificar se o sistema está online")
def health():
    return {
        "status": "online",
        "service": "Cortes Inteligentes com IA",
        "version": "3.4.1",
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_downloaded_video_mb": MAX_DOWNLOADED_VIDEO_MB,
        "max_video_minutes": MAX_VIDEO_MINUTES,
    }

def baixar_video(url: str, destination: Path) -> str:
    info_cmd = [
        "yt-dlp", "--no-playlist", "--dump-single-json",
        "--skip-download", url,
    ]
    info_result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=60)
    if info_result.returncode != 0:
        raise RuntimeError(f"Não foi possível acessar o vídeo: {info_result.stderr[-1000:]}")

    try:
        info = json.loads(info_result.stdout)
        if info.get("is_live") is True or info.get("live_status") == "is_live":
            raise RuntimeError(
                "Este vídeo ainda está AO VIVO. Aguarde a transmissão terminar "
                "ou envie um arquivo gravado para gerar os cortes."
            )
    except json.JSONDecodeError:
        pass

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--newline",
        "--socket-timeout", "30",
        "--retries", "3",
        "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/best[height<=1080]",
        "--merge-output-format", "mp4",
        "--force-overwrites",
        "-o", str(destination),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Erro ao baixar o vídeo: {result.stderr[-1200:]}")
    if not destination.exists():
        raise RuntimeError("O download terminou, mas o arquivo não foi encontrado.")
    return str(destination)

def _processar_url_job(job_id: str, url: str, destination: str, max_cortes: int, min_duracao: int, max_duracao: int):
    try:
        _update_job(
            job_id,
            status="downloading",
            progress=3,
            message="Baixando o vídeo. Em vídeos longos isso pode levar alguns minutos.",
        )
        baixar_video(url, Path(destination))
        _processar_job(job_id, destination, max_cortes, min_duracao, max_duracao, "url")
    except Exception as exc:
        _update_job(
            job_id,
            status="error",
            progress=100,
            message="Falha no processamento",
            error=str(exc),
        )

def _processar_job(job_id: str, video_path: str, max_cortes: int, min_duracao: int, max_duracao: int, source_type: str = "upload"):
    output_dir = CLIPS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _update_job(job_id, status="validating", progress=5, message="Validando vídeo")
        _validate_video_limits(video_path, source_type)

        _update_job(job_id, status="transcribing", progress=15, message="Transcrevendo vídeo")
        transcricao = transcrever_video(video_path)

        _update_job(job_id, status="analyzing", progress=55, message="Analisando melhores momentos")
        melhores = analisar_video_com_gemini(
            transcricao["segments"],
            transcricao["duration"],
            max_cortes=max_cortes,
            min_duracao=min_duracao,
            max_duracao=max_duracao,
        )

        if not melhores:
            raise RuntimeError("A IA não encontrou cortes válidos dentro do contexto.")

        _update_job(job_id, status="cutting", progress=80, message="Gerando cortes")
        arquivos = cortar_segmentos(video_path, str(output_dir), melhores)

        result = {
            "duration": transcricao["duration"],
            "language": transcricao.get("language"),
            "clips": arquivos,
            "analysis": melhores,
        }
        _update_job(
            job_id,
            status="done",
            progress=100,
            message="Processamento concluído",
            result=result,
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="error",
            progress=100,
            message="Falha no processamento",
            error=str(exc),
        )

def _create_job(source_type: str, source: str, video_path: str, max_cortes: int, min_duracao: int, max_duracao: int):
    _validate_durations(min_duracao, max_duracao)
    job_id = str(uuid.uuid4())
    now = time.time()
    _save_job(job_id, {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Na fila",
        "source_type": source_type,
        "source": source,
        "video_path": video_path,
        "created_at": now,
        "updated_at": now,
        "result": None,
        "error": None,
    })
    thread = threading.Thread(
        target=_processar_job,
        args=(job_id, video_path, max_cortes, min_duracao, max_duracao, source_type),
        daemon=True,
    )
    thread.start()
    return _load_job(job_id)

@app.post("/process-url")
def process_url(data: VideoRequest):
    _validate_durations(data.min_duracao, data.max_duracao)
    job_id = str(uuid.uuid4())
    destination = DOWNLOAD_DIR / f"{job_id}.mp4"
    now = time.time()

    _save_job(job_id, {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Preparando download",
        "source_type": "url",
        "source": str(data.url),
        "video_path": str(destination),
        "created_at": now,
        "updated_at": now,
        "result": None,
        "error": None,
    })

    thread = threading.Thread(
        target=_processar_url_job,
        args=(
            job_id,
            str(data.url),
            str(destination),
            data.max_cortes,
            data.min_duracao,
            data.max_duracao,
        ),
        daemon=True,
    )
    thread.start()
    return _load_job(job_id)

@app.post("/process-upload")
async def process_upload(
    file: UploadFile = File(...),
    max_cortes: int = Query(default=3, ge=1, le=10),
    min_duracao: int = Query(default=45, ge=15, le=600),
    max_duracao: int = Query(default=180, ge=20, le=900),
):
    _validate_durations(min_duracao, max_duracao)

    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        raise HTTPException(status_code=400, detail="Formato de vídeo não suportado.")

    destination = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    bytes_written = 0
    limit_bytes = MAX_UPLOAD_MB * 1024 * 1024

    try:
        with destination.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > limit_bytes:
                    buffer.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Arquivo excede o limite de {MAX_UPLOAD_MB} MB."
                    )
                buffer.write(chunk)
    finally:
        await file.close()

    return _create_job(
        "upload",
        file.filename or destination.name,
        str(destination),
        max_cortes,
        min_duracao,
        max_duracao,
    )

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    return _load_job(job_id)

@app.get("/jobs/{job_id}/clips/{clip_index}")
def download_clip(job_id: str, clip_index: int):
    job = _load_job(job_id)
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail="O processamento ainda não terminou.")

    clips = (job.get("result") or {}).get("clips") or []
    if clip_index < 1 or clip_index > len(clips):
        raise HTTPException(status_code=404, detail="Corte não encontrado.")

    clip = clips[clip_index - 1]
    path = Path(clip["file"]).resolve()
    expected_root = (CLIPS_DIR / job_id).resolve()
    if expected_root not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Arquivo do corte não encontrado.")

    return FileResponse(
        path=str(path),
        media_type="video/mp4",
        filename=path.name,
    )

@app.post("/maintenance/cleanup")
def cleanup():
    _cleanup_old_files()
    return {"status": "ok", "retention_hours": RETENTION_HOURS}
