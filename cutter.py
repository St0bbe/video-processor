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
        blocks.append(f"{i}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")



def _ass_time(t):
    cs = max(0, round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _write_karaoke(path, segments, clip_start, clip_end, font, margin):
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Karaoke,{font},58,&H0000D7FF,&H00FFFFFF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,3,1,2,65,65,{margin},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for seg in segments or []:
        words = []
        for w in seg.get("words") or []:
            start, end = float(w["start"]), float(w["end"])
            token = str(w["word"]).strip().replace("{", "(").replace("}", ")").replace("\\", "")
            if token and end > clip_start and start < clip_end:
                words.append((max(0, start - clip_start), min(clip_end, end) - clip_start, token))
        for index in range(0, len(words), 5):
            group = words[index:index + 5]
            if not group:
                continue
            start, end = group[0][0], max(w[1] for w in group)
            if end <= start:
                continue
            parts = []
            cursor = start
            for ws, we, token in group:
                if ws > cursor:
                    parts.append(r"{\k" + str(max(1, round((ws - cursor) * 100))) + "}")
                parts.append(r"{\kf" + str(max(1, round((we - max(cursor, ws)) * 100))) + "}" + token + " ")
                cursor = max(cursor, we)
            events.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Karaoke,,0,0,0,,{''.join(parts).strip()}\n")
    if events:
        path.write_text(header + "".join(events), encoding="utf-8")
    return bool(events)


def _ffmpeg_filter_path(path: Path) -> str:
    value = path.resolve().as_posix()
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    return value


def cortar_segmentos(
    input_path: str,
    output_dir: str,
    segmentos,
    transcript_segments=None,
    logo_path: str | None = None,
    vertical: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    resultados = []
    subtitles_enabled = os.getenv("BURN_SUBTITLES", "true").lower() not in {"0", "false", "no", "off"}
    font_name = os.getenv("SUBTITLE_FONT", "Arial")
    font_size = int(os.getenv("SUBTITLE_FONT_SIZE", "22"))
    margin_v = int(os.getenv("SUBTITLE_MARGIN_V", "110" if vertical else "42"))
    logo_file = Path(logo_path) if logo_path else None
    has_logo = bool(logo_file and logo_file.exists())

    for indice, segmento in enumerate(segmentos, start=1):
        inicio = max(0.0, float(segmento["start_seconds"]) - 0.08)
        fim = float(segmento["end_seconds"]) + 0.12
        duracao = max(0.1, fim - inicio)
        nome = f"{indice:02d}_{_slug(segmento.get('title', 'corte'))}.mp4"
        clip_path = Path(output_dir) / nome
        srt_path = Path(output_dir) / f"{indice:02d}_{_slug(segmento.get('title', 'corte'))}.srt"

        subtitle_items = _subtitle_segments(transcript_segments, inicio, fim)
        ass_path = srt_path.with_suffix(".ass")
        karaoke = False
        if subtitles_enabled and subtitle_items:
            if os.getenv("SUBTITLE_KARAOKE", "true").lower() not in {"false", "0", "off"}:
                karaoke = _write_karaoke(ass_path, transcript_segments, inicio, fim, font_name, margin_v)
            if not karaoke:
                _write_srt(srt_path, subtitle_items)

        inputs = ["-ss", f"{inicio:.3f}", "-i", input_path]
        if has_logo:
            inputs += ["-i", str(logo_file)]

        filters = []
        base = "[0:v]"
        if vertical:
            # TikTok/Reels 9:16. Fundo preenchido sem esticar o vídeo e conteúdo
            # principal centralizado, preservando toda a imagem original.
            filters += [
                "[0:v]split=2[bgsrc][fgsrc]",
                "[bgsrc]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:10[bg]",
                "[fgsrc]scale=1080:1920:force_original_aspect_ratio=decrease[fg]",
                "[bg][fg]overlay=(W-w)/2:(H-h)/2[canvas]",
            ]
            base = "[canvas]"

        if subtitles_enabled and subtitle_items:
            subtitle_file = _ffmpeg_filter_path(ass_path if karaoke else srt_path)
            style = (
                f"FontName={font_name},FontSize={font_size},"
                "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                "BorderStyle=1,Outline=2,Shadow=1,Alignment=2,"
                f"MarginV={margin_v}"
            )
            if karaoke:
                filters.append(f"{base}subtitles='{subtitle_file}'[subbed]")
            else:
                filters.append(f"{base}subtitles='{subtitle_file}':force_style='{style}'[subbed]")
            base = "[subbed]"

        if has_logo:
            logo_input = "[1:v]"
            filters.append(f"{logo_input}scale='min(260,iw)':-1[logo]")
            filters.append(f"{base}[logo]overlay=W-w-36:H-h-70:format=auto[outv]")
            base = "[outv]"

        if not filters:
            filters.append("[0:v]null[outv]")
            base = "[outv]"

        cmd = ["ffmpeg", "-y", *inputs, "-t", f"{duracao:.3f}", "-filter_complex", ";".join(filters),
               "-map", base, "-map", "0:a?", "-c:v", "libx264",
               "-preset", os.getenv("FFMPEG_PRESET", "veryfast"),
               "-crf", os.getenv("FFMPEG_CRF", "20"), "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
               "-avoid_negative_ts", "make_zero", str(clip_path)]

        result = subprocess.run(cmd, capture_output=True, text=True)
        srt_path.unlink(missing_ok=True)
        ass_path.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg falhou no corte {indice}: {result.stderr[-1600:]}")

        resultados.append({
            "index": indice,
            "file": str(clip_path),
            "download_path": f"/jobs/{{job_id}}/clips/{indice}",
            "title": segmento.get("title"),
            "start": round(inicio, 3),
            "end": round(fim, 3),
            "duration": round(duracao, 3),
            "subtitles": bool(subtitles_enabled and subtitle_items),
            "vertical": vertical,
            "logo": has_logo,
            "virality_score": segmento.get("virality_score"),
            "context_score": segmento.get("context_score"),
            "reason": segmento.get("reason"),
        })

    return resultados
