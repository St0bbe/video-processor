import os
import subprocess
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is None:
        model_size = os.getenv("WHISPER_MODEL", "small")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv(
            "WHISPER_COMPUTE_TYPE",
            "int8" if device == "cpu" else "float16",
        )
        _MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _MODEL

def _duracao_video(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Não foi possível descobrir a duração do vídeo.")
    return float(result.stdout.strip())

def _extrair_audio(video_path: str, audio_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Erro ao extrair áudio: {result.stderr[-1000:]}")

def transcrever_video(video_path: str):
    model = _get_model()
    duration = _duracao_video(video_path)

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = str(Path(tmp) / "audio.wav")
        _extrair_audio(video_path, audio_path)

        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            word_timestamps=False,
        )

        items = []
        for seg in segments:
            texto = seg.text.strip()
            if texto:
                items.append({
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": texto,
                })

    return {
        "duration": duration,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "segments": items,
    }
