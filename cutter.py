import os
import re
import subprocess

def _slug(texto: str) -> str:
    texto = re.sub(r"[^a-zA-Z0-9_-]+", "_", texto.strip())
    return texto.strip("_")[:60] or "corte"

def cortar_segmentos(input_path: str, output_dir: str, segmentos):
    os.makedirs(output_dir, exist_ok=True)
    resultados = []

    for indice, segmento in enumerate(segmentos, start=1):
        inicio = float(segmento["start_seconds"])
        fim = float(segmento["end_seconds"])
        duracao = fim - inicio

        nome = f"{indice:02d}_{_slug(segmento.get('title', 'corte'))}.mp4"
        clip_path = os.path.join(output_dir, nome)

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{inicio:.3f}",
            "-i", input_path,
            "-t", f"{duracao:.3f}",
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            clip_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg falhou no corte {indice}: {result.stderr[-1000:]}")

        resultados.append({
            "file": clip_path,
            "title": segmento.get("title"),
            "start": inicio,
            "end": fim,
            "duration": round(duracao, 3),
            "virality_score": segmento.get("virality_score"),
            "reason": segmento.get("reason"),
        })

    return resultados
