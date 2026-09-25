import os
import re
import subprocess
from pathlib import Path

def _slug(texto: str) -> str:
    texto = re.sub(r"[^a-zA-Z0-9_-]+", "_", texto.strip())
    return texto.strip("_")[:60] or "corte"

def cortar_segmentos(input_path: str, output_dir: str, segmentos):
    os.makedirs(output_dir, exist_ok=True)
    resultados = []

    for indice, segmento in enumerate(segmentos, start=1):
        inicio = max(0.0, float(segmento["start_seconds"]) - 0.08)
        fim = float(segmento["end_seconds"]) + 0.12
        duracao = max(0.1, fim - inicio)

        nome = f"{indice:02d}_{_slug(segmento.get('title', 'corte'))}.mp4"
        clip_path = Path(output_dir) / nome

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{inicio:.3f}",
            "-i", input_path,
            "-t", f"{duracao:.3f}",
            "-map", "0:v:0",
            "-map", "0:a?",
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

        resultados.append({
            "index": indice,
            "file": str(clip_path),
            "download_path": f"/jobs/{{job_id}}/clips/{indice}",
            "title": segmento.get("title"),
            "start": round(inicio, 3),
            "end": round(fim, 3),
            "duration": round(duracao, 3),
            "virality_score": segmento.get("virality_score"),
            "context_score": segmento.get("context_score"),
            "reason": segmento.get("reason"),
        })

    return resultados
