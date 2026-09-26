import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

from faster_whisper import WhisperModel

_MODEL = None
_MODEL_LOCK = threading.Lock()


def _get_model(progress_callback: Optional[Callable[[int, str], None]] = None):
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        model_size = os.getenv("WHISPER_MODEL", "small")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv(
            "WHISPER_COMPUTE_TYPE",
            "int8" if device == "cpu" else "float16",
        )

        if progress_callback:
            progress_callback(18, f"Carregando Whisper {model_size}. Na primeira execução o modelo pode ser baixado.")

        _MODEL = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
    return _MODEL


def preload_model():
    """Carrega/baixa o Whisper antes do primeiro vídeo."""
    return _get_model()


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


def transcrever_video(
    video_path: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
):
    duration = _duracao_video(video_path)

    if progress_callback:
        progress_callback(16, "Preparando o reconhecimento de voz")

    model = _get_model(progress_callback)

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = str(Path(tmp) / "audio.wav")

        if progress_callback:
            progress_callback(20, "Extraindo áudio do vídeo")
        _extrair_audio(video_path, audio_path)

        if progress_callback:
            progress_callback(23, "Transcrevendo áudio com Whisper")

        # Em CPU, beam_size=1 é muito mais rápido para vídeos longos.
        # Timestamps por segmento são suficientes para a IA escolher cortes
        # contextualizados e evitam o custo elevado de word_timestamps.
        device = os.getenv("WHISPER_DEVICE", "cpu")
        beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "1" if device == "cpu" else "3"))
        word_timestamps = os.getenv("WHISPER_WORD_TIMESTAMPS", "false").lower() == "true"

        segments, info = model.transcribe(
            audio_path,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=word_timestamps,
            condition_on_previous_text=False,
        )

        items = []
        last_progress = 23

        # faster-whisper só executa de fato enquanto os segmentos são iterados.
        for seg in segments:
            texto = seg.text.strip()
            if not texto:
                continue

            words = []
            if word_timestamps:
                for word in seg.words or []:
                    if word.start is None or word.end is None:
                        continue
                    words.append({
                        "start": float(word.start),
                        "end": float(word.end),
                        "word": word.word,
                        "probability": getattr(word, "probability", None),
                    })

            items.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": texto,
                "words": words,
            })

            if duration > 0 and progress_callback:
                ratio = min(float(seg.end) / duration, 1.0)
                current = 23 + int(ratio * 29)  # 23% -> 52%
                if current >= last_progress + 2:
                    minutos = int(seg.end // 60)
                    total_minutos = max(1, int(duration // 60))
                    progress_callback(
                        current,
                        f"Transcrevendo: {minutos} de {total_minutos} min analisados",
                    )
                    last_progress = current

    if not items:
        raise RuntimeError(
            "O Whisper não encontrou fala suficiente no vídeo para gerar cortes."
        )

    if progress_callback:
        progress_callback(53, "Transcrição concluída")

    return {
        "duration": duration,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "segments": items,
    }
