import os
import re
import subprocess
from pathlib import Path


def _slug(texto: str) -> str:
    texto = re.sub(r"[^a-zA-Z0-9_-]+", "_", texto.strip())
    return texto.strip("_")[:60] or "corte"


def _srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _subtitle_segments(transcript_segments, clip_start: float, clip_end: float):
    items = []
    for seg in transcript_segments or []:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        text = str(seg.get("text", "")).strip()
        if not text or end <= clip_start or start >= clip_end:
            continue
        local_start = max(0.0, start - clip_start)
        local_end = min(clip_end, end) - clip_start
        if local_end > local_start:
            items.append((local_start, local_end, text))
    return items


def _write_srt(path: Path, items):
    blocks = []
    for i, (start, end, text) in enumerate(items, 1):
        blocks.append(
            f"{i}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


def _ffmpeg_subtitle_path(path: Path) -> str:
    # FFmpeg/libass no Windows precisa de barras normais e ':' escapado.
    value = path.resolve().as_posix()
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    return value


def cortar_segmentos(input_path: str, output_dir: str, segmentos, transcript_segments=None):
    os.makedirs(output_dir, exist_ok=True)
    resultados = []
    subtitles_enabled = os.getenv("BURN_SUBTITLES", "true").lower() not in {"0", "false", "no", "off"}

    font_name = os.getenv("SUBTITLE_FONT", "Arial")
    font_size = int(os.getenv("SUBTITLE_FONT_SIZE", "22"))
    margin_v = int(os.getenv("SUBTITLE_MARGIN_V", "42"))

    for indice, segmento in enumerate(segmentos, start=1):
        inicio = max(0.0, float(segmento["start_seconds"]) - 0.08)
        fim = float(segmento["end_seconds"]) + 0.12
        duracao = max(0.1, fim - inicio)

        nome = f"{indice:02d}_{_slug(segmento.get('title', 'corte'))}.mp4"
        clip_path = Path(output_dir) / nome
        srt_path = Path(output_dir) / f"{indice:02d}_{_slug(segmento.get('title', 'corte'))}.srt"

        video_filters = []
        subtitle_items = _subtitle_segments(transcript_segments, inicio, fim)
        if subtitles_enabled and subtitle_items:
            _write_srt(srt_path, subtitle_items)
            subtitle_file = _ffmpeg_subtitle_path(srt_path)
            style = (
                f"FontName={font_name},FontSize={font_size},"
                "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                "BorderStyle=1,Outline=2,Shadow=1,Alignment=2,"
                f"MarginV={margin_v}"
            )
            video_filters = ["-vf", f"subtitles='{subtitle_file}':force_style='{style}'"]

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{inicio:.3f}",
            "-i", input_path,
            "-t", f"{duracao:.3f}",
            "-map", "0:v:0",
            "-map", "0:a?",
            *video_filters,
            "-c:v", "libx264",
            "-preset", os.getenv("FFMPEG_PRESET", "veryfast"),
            "-crf", os.getenv("FFMPEG_CRF", "20"),
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            str(clip_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg falhou no corte {indice}: {result.stderr[-1200:]}")

        # A legenda já está queimada no MP4; o SRT é apenas temporário.
        srt_path.unlink(missing_ok=True)

        resultados.append({
            "index": indice,
            "file": str(clip_path),
            "download_path": f"/jobs/{{job_id}}/clips/{indice}",
            "title": segmento.get("title"),
            "start": round(inicio, 3),
            "end": round(fim, 3),
            "duration": round(duracao, 3),
            "subtitles": bool(subtitles_enabled and subtitle_items),
            "virality_score": segmento.get("virality_score"),
            "context_score": segmento.get("context_score"),
            "reason": segmento.get("reason"),
        })

    return resultados
