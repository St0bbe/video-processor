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
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field, HttpUrl
from dotenv import load_dotenv

from transcriber import transcrever_video, preload_model
from ai_editor import analisar_video_com_gemini
from cutter import cortar_segmentos
from tiktok_integration import TikTokIntegration, TikTokIntegrationError

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DOWNLOAD_DIR = BASE_DIR / "downloads"
UPLOAD_DIR = BASE_DIR / "uploads"
CLIPS_DIR = BASE_DIR / "clips"
JOBS_DIR = BASE_DIR / "jobs"

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "4096"))
MAX_DOWNLOADED_VIDEO_MB = int(os.getenv("MAX_DOWNLOADED_VIDEO_MB", "8192"))
MAX_VIDEO_MINUTES = int(os.getenv("MAX_VIDEO_MINUTES", "240"))
RETENTION_HOURS = int(os.getenv("RETENTION_HOURS", "48"))
MIN_FREE_DISK_GB = float(os.getenv("MIN_FREE_DISK_GB", "8"))
TIKTOK = TikTokIntegration(BASE_DIR)

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
    version="3.9.0",
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
    logo_path: Optional[str] = None

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

    # Pré-carrega o Whisper sem bloquear a inicialização da interface.
    # Na primeira execução o modelo é baixado e fica em cache para os próximos vídeos.
    def _warmup_whisper():
        try:
            preload_model()
            print("Whisper carregado e pronto para transcrever.")
        except Exception as exc:
            print(f"Aviso: não foi possível pré-carregar o Whisper: {exc}")

    threading.Thread(target=_warmup_whisper, daemon=True).start()

@app.get("/termos", response_class=HTMLResponse, include_in_schema=False)
def termos():
    page = BASE_DIR / "static" / "termos.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Termos de Serviço não encontrados.")
    return HTMLResponse(page.read_text(encoding="utf-8"))

@app.get("/privacidade", response_class=HTMLResponse, include_in_schema=False)
def privacidade():
    page = BASE_DIR / "static" / "privacidade.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Política de Privacidade não encontrada.")
    return HTMLResponse(page.read_text(encoding="utf-8"))

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def interface():
    page = BASE_DIR / "static" / "index.html"
    if not page.exists():
        return HTMLResponse(
            "<h1>Interface não encontrada.</h1><p>Verifique static/index.html.</p>",
            status_code=500,
        )
    return HTMLResponse(page.read_text(encoding="utf-8"))

class TikTokPublishRequest(BaseModel):
    job_id: str
    clip_index: int = Field(ge=1)
    title: str = Field(default="", max_length=2200)
    privacy_level: str
    disable_comment: bool = False
    disable_duet: bool = False
    disable_stitch: bool = False
    is_aigc: bool = False


@app.get("/auth/tiktok", include_in_schema=False)
def tiktok_login():
    try:
        return RedirectResponse(TIKTOK.authorization_url())
    except TikTokIntegrationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/auth/tiktok/callback", include_in_schema=False)
def tiktok_callback(
    code: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None),
    error: Optional[str] = Query(default=None),
    error_description: Optional[str] = Query(default=None),
):
    if error:
        message = error_description or error
        return RedirectResponse("/?tiktok=error&message=" + TIKTOK.quote(message))
    if not code or not state:
        return RedirectResponse("/?tiktok=error&message=Resposta%20OAuth%20incompleta")
    try:
        TIKTOK.exchange_code(code, state)
        return RedirectResponse("/?tiktok=connected")
    except TikTokIntegrationError as exc:
        return RedirectResponse("/?tiktok=error&message=" + TIKTOK.quote(str(exc)))


@app.get("/api/tiktok/status", tags=["TikTok"])
def tiktok_status():
    return TIKTOK.status()


@app.post("/api/tiktok/disconnect", tags=["TikTok"])
def tiktok_disconnect():
    TIKTOK.disconnect()
    return {"connected": False}


@app.post("/api/tiktok/creator-info", tags=["TikTok"])
def tiktok_creator_info():
    try:
        return TIKTOK.creator_info()
    except TikTokIntegrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/tiktok/publish", tags=["TikTok"])
def tiktok_publish(payload: TikTokPublishRequest):
    job = _load_job(payload.job_id)
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail="O processamento ainda não terminou.")
    clips = (job.get("result") or {}).get("clips") or []
    if payload.clip_index > len(clips):
        raise HTTPException(status_code=404, detail="Corte não encontrado.")
    clip_path = Path(clips[payload.clip_index - 1]["file"]).resolve()
    expected_root = (CLIPS_DIR / payload.job_id).resolve()
    if expected_root not in clip_path.parents or not clip_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo do corte não encontrado.")
    try:
        return TIKTOK.publish_video(
            clip_path,
            title=payload.title,
            privacy_level=payload.privacy_level,
            disable_comment=payload.disable_comment,
            disable_duet=payload.disable_duet,
            disable_stitch=payload.disable_stitch,
            is_aigc=payload.is_aigc,
        )
    except TikTokIntegrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/tiktok/publish-status/{publish_id}", tags=["TikTok"])
def tiktok_publish_status(publish_id: str):
    try:
        return TIKTOK.publish_status(publish_id)
    except TikTokIntegrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/status", tags=["⚙️ Sistema"], summary="Verificar se o sistema está online")
def health():
    return {
        "status": "online",
        "service": "Cortes Inteligentes com IA",
        "version": "3.9.0",
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_downloaded_video_mb": MAX_DOWNLOADED_VIDEO_MB,
        "max_video_minutes": MAX_VIDEO_MINUTES,
    }

def _yt_dlp_js_args():
    """Configura um runtime JS suportado pelo yt-dlp para os desafios do YouTube."""
    if shutil.which("deno"):
        return ["--js-runtimes", f"deno:{shutil.which('deno')}"]
    if shutil.which("node"):
        return ["--js-runtimes", f"node:{shutil.which('node')}"]
    if shutil.which("bun"):
        return ["--js-runtimes", f"bun:{shutil.which('bun')}"]
    if shutil.which("qjs"):
        return ["--js-runtimes", f"quickjs:{shutil.which('qjs')}"]
    return []


def _yt_dlp_cookie_args():
    """Usa cookies locais sem registrar ou expor seu conteúdo."""
    configured = os.getenv("YOUTUBE_COOKIES_FILE", "").strip()
    if not configured:
        return []
    cookie_path = Path(configured).expanduser()
    if not cookie_path.is_absolute():
        cookie_path = BASE_DIR / cookie_path
    if not cookie_path.is_file():
        raise RuntimeError("Arquivo de cookies do YouTube não encontrado. Confira YOUTUBE_COOKIES_FILE.")
    return ["--cookies", str(cookie_path)]


def _friendly_ytdlp_error(stderr: str) -> str:
    text = stderr or ""
    lower = text.lower()
    if "sign in to confirm" in lower or "not a bot" in lower:
        return ("O YouTube solicitou autenticação para liberar o vídeo. "
                "Configure ou atualize YOUTUBE_COOKIES_FILE no servidor.")
    if "no supported javascript runtime" in lower:
        return (
            "O YouTube exige um runtime JavaScript para liberar este vídeo. "
            "Instale Node.js 22+ ou Deno 2.3+ e reinicie o aplicativo."
        )
    if "403" in lower or "forbidden" in lower:
        return (
            "O YouTube bloqueou o download deste vídeo (erro 403). "
            "Atualize as dependências do projeto e confirme que Node.js 22+ ou Deno 2.3+ está instalado."
        )
    return f"Não foi possível baixar o vídeo. Detalhes: {text[-700:]}"


def _ensure_disk_space(path: Path, min_free_gb: float = MIN_FREE_DISK_GB):
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Espaço insuficiente no disco. Disponível: {free_gb:.1f} GB. "
            f"Libere pelo menos {min_free_gb:.0f} GB antes de processar o vídeo."
        )


def _cleanup_download_artifacts(destination: Path, keep_final: bool = False):
    """Remove restos .part/.temp e formatos separados criados pelo yt-dlp."""
    stem = destination.stem
    for item in destination.parent.glob(f"{stem}*"):
        try:
            if keep_final and item.resolve() == destination.resolve():
                continue
            if item.is_file():
                item.unlink(missing_ok=True)
        except OSError:
            pass


def baixar_video(url: str, destination: Path) -> str:
    _ensure_disk_space(destination.parent)
    js_args = _yt_dlp_js_args()
    cookie_args = _yt_dlp_cookie_args()
    info_cmd = [
        "yt-dlp", "--no-playlist", "--dump-single-json",
        "--skip-download", *js_args, *cookie_args, url,
    ]
    info_result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=60)
    if info_result.returncode != 0:
        raise RuntimeError(_friendly_ytdlp_error(info_result.stderr))

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
        *js_args,
        *cookie_args,
        "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/best[height<=720]",
        "--merge-output-format", "mp4",
        "--force-overwrites",
        "-o", str(destination),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _cleanup_download_artifacts(destination, keep_final=False)
        raise RuntimeError(_friendly_ytdlp_error(result.stderr))
    if not destination.exists():
        raise RuntimeError("O download terminou, mas o arquivo não foi encontrado.")
    return str(destination)

def _processar_url_job(job_id: str, url: str, destination: str, max_cortes: int, min_duracao: int, max_duracao: int, logo_path: str | None = None):
    try:
        _update_job(
            job_id,
            status="downloading",
            progress=3,
            message="Baixando o vídeo. Em vídeos longos isso pode levar alguns minutos.",
        )
        baixar_video(url, Path(destination))
        _processar_job(job_id, destination, max_cortes, min_duracao, max_duracao, "url", logo_path)
    except Exception as exc:
        _cleanup_download_artifacts(Path(destination), keep_final=False)
        _update_job(
            job_id,
            status="error",
            progress=100,
            message="Falha no processamento",
            error=str(exc),
        )

def _processar_job(job_id: str, video_path: str, max_cortes: int, min_duracao: int, max_duracao: int, source_type: str = "upload", logo_path: str | None = None):
    output_dir = CLIPS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _update_job(job_id, status="validating", progress=5, message="Validando vídeo")
        _validate_video_limits(video_path, source_type)

        _update_job(job_id, status="transcribing", progress=15, message="Preparando transcrição")

        def transcription_progress(progress: int, message: str):
            _update_job(
                job_id,
                status="transcribing",
                progress=progress,
                message=message,
            )

        transcricao = transcrever_video(
            video_path,
            progress_callback=transcription_progress,
        )

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

        _update_job(job_id, status="cutting", progress=80, message="Gerando cortes e adicionando legendas")
        arquivos = cortar_segmentos(
            video_path,
            str(output_dir),
            melhores,
            transcript_segments=transcricao["segments"],
            logo_path=logo_path,
            vertical=True,
        )

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

        # Para links, os cortes finais ficam em clips/ e o vídeo original
        # baixado não precisa continuar ocupando vários GB.
        if source_type == "url":
            _cleanup_download_artifacts(Path(video_path), keep_final=False)
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
        args=(job_id, video_path, max_cortes, min_duracao, max_duracao, source_type, None),
        daemon=True,
    )
    thread.start()
    return _load_job(job_id)

@app.post("/process-url")
async def process_url(
    url: str = Query(...),
    max_cortes: int = Query(default=3, ge=1, le=10),
    min_duracao: int = Query(default=45, ge=15, le=600),
    max_duracao: int = Query(default=180, ge=20, le=900),
    logo: Optional[UploadFile] = File(default=None),
):
    _validate_durations(min_duracao, max_duracao)
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Link de vídeo inválido.")

    job_id = str(uuid.uuid4())
    destination = DOWNLOAD_DIR / f"{job_id}.mp4"
    logo_path = None

    if logo and logo.filename:
        logo_suffix = Path(logo.filename).suffix.lower()
        if logo_suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            await logo.close()
            raise HTTPException(status_code=400, detail="Logo deve ser PNG, JPG, JPEG ou WEBP.")
        logo_destination = UPLOAD_DIR / f"{job_id}_logo{logo_suffix}"
        try:
            logo_data = await logo.read()
            if len(logo_data) > 10 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="A logo deve ter no máximo 10 MB.")
            logo_destination.write_bytes(logo_data)
            logo_path = str(logo_destination)
        finally:
            await logo.close()

    now = time.time()
    _save_job(job_id, {
        "job_id": job_id, "status": "queued", "progress": 0,
        "message": "Preparando download", "source_type": "url",
        "source": url, "video_path": str(destination), "logo_path": logo_path,
        "created_at": now, "updated_at": now, "result": None, "error": None,
    })
    thread = threading.Thread(
        target=_processar_url_job,
        args=(job_id, url, str(destination), max_cortes, min_duracao, max_duracao, logo_path),
        daemon=True,
    )
    thread.start()
    return _load_job(job_id)

@app.post("/process-upload")
async def process_upload(
    file: UploadFile = File(...),
    logo: Optional[UploadFile] = File(default=None),
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

    logo_path = None
    if logo and logo.filename:
        logo_suffix = Path(logo.filename).suffix.lower()
        if logo_suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            destination.unlink(missing_ok=True)
            await logo.close()
            raise HTTPException(status_code=400, detail="Logo deve ser PNG, JPG, JPEG ou WEBP.")
        logo_destination = UPLOAD_DIR / f"{uuid.uuid4()}_logo{logo_suffix}"
        try:
            logo_data = await logo.read()
            if len(logo_data) > 10 * 1024 * 1024:
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="A logo deve ter no máximo 10 MB.")
            logo_destination.write_bytes(logo_data)
            logo_path = str(logo_destination)
        finally:
            await logo.close()

    _validate_durations(min_duracao, max_duracao)
    job_id = str(uuid.uuid4())
    now = time.time()
    _save_job(job_id, {
        "job_id": job_id, "status": "queued", "progress": 0, "message": "Na fila",
        "source_type": "upload", "source": file.filename or destination.name,
        "video_path": str(destination), "logo_path": logo_path,
        "created_at": now, "updated_at": now, "result": None, "error": None,
    })
    thread = threading.Thread(
        target=_processar_job,
        args=(job_id, str(destination), max_cortes, min_duracao, max_duracao, "upload", logo_path),
        daemon=True,
    )
    thread.start()
    return _load_job(job_id)

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
