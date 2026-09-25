import os
import json
import math
from typing import TypedDict

import google.generativeai as genai

class ViralSegment(TypedDict):
    start_time: str
    end_time: str
    title: str
    virality_score: int
    reason: str

def formatar_tempo(segundos: float) -> str:
    segundos = max(0, int(segundos))
    h = segundos // 3600
    m = (segundos % 3600) // 60
    s = segundos % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def tempo_para_segundos(valor: str) -> float:
    partes = [float(p) for p in valor.strip().split(":")]
    if len(partes) == 2:
        return partes[0] * 60 + partes[1]
    if len(partes) == 3:
        return partes[0] * 3600 + partes[1] * 60 + partes[2]
    raise ValueError(f"Timestamp inválido: {valor}")

def _configurar_modelo():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Defina a variável de ambiente GEMINI_API_KEY.")

    genai.configure(api_key=api_key)

    modelos = [
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "gemini-2.0-flash",
    ]

    ultimo_erro = None
    for nome in modelos:
        try:
            model = genai.GenerativeModel(nome)
            return model
        except Exception as exc:
            ultimo_erro = exc

    raise RuntimeError(f"Nenhum modelo Gemini disponível: {ultimo_erro}")

def _ajustar_para_limites(segmento, duration_video, min_duracao, max_duracao):
    inicio = max(0.0, tempo_para_segundos(segmento["start_time"]))
    fim = min(float(duration_video), tempo_para_segundos(segmento["end_time"]))

    if fim <= inicio:
        return None

    duracao = fim - inicio
    if duracao < min_duracao or duracao > max_duracao:
        return None

    segmento["start_seconds"] = round(inicio, 3)
    segmento["end_seconds"] = round(fim, 3)
    segmento["duration"] = round(duracao, 3)
    return segmento

def analisar_video_com_gemini(
    segments,
    duration_video,
    max_cortes=3,
    min_duracao=45,
    max_duracao=180,
):
    if not segments:
        return []

    transcript = []
    for seg in segments:
        texto = seg.get("text", "").strip()
        if not texto:
            continue
        inicio = formatar_tempo(seg.get("start", 0))
        fim = formatar_tempo(seg.get("end", seg.get("start", 0)))
        transcript.append(f"[{inicio} - {fim}] {texto}")

    transcript_text = "\n".join(transcript)

    model = _configurar_modelo()

    prompt = f"""
Você é um editor profissional de vídeos curtos para TikTok, Reels e Shorts.

Analise a transcrição completa abaixo e selecione até {max_cortes} melhores momentos.

OBJETIVO PRINCIPAL:
Escolher trechos que façam sentido sozinhos e preservem o contexto. Não escolha apenas frases chamativas.
Cada corte precisa ter uma abertura compreensível, desenvolvimento e conclusão natural.

REGRAS:
1. Cada corte deve durar entre {min_duracao} e {max_duracao} segundos.
2. Comece no início de uma frase ou ideia, nunca no meio.
3. Termine somente depois que o raciocínio estiver concluído.
4. Evite trechos que dependam fortemente de algo dito antes e que deixem o espectador perdido.
5. Priorize histórias, opiniões fortes, revelações, explicações claras, conflitos, humor, emoção e informações surpreendentes.
6. Não invente conteúdo.
7. Não gere cortes sobrepostos, a menos que seja absolutamente necessário.
8. Dê uma nota de potencial entre 0 e 100.
9. Use timestamps EXATOS disponíveis na transcrição.
10. Retorne somente JSON válido.

FORMATO:
[
  {{
    "start_time": "00:02:15",
    "end_time": "00:03:40",
    "title": "Título curto e atraente",
    "virality_score": 87,
    "reason": "Explique em uma frase por que este trecho funciona sozinho e tem potencial."
  }}
]

DURAÇÃO TOTAL DO VÍDEO: {formatar_tempo(duration_video)}

TRANSCRIÇÃO:
{transcript_text}
"""

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json"
        ),
    )

    dados = json.loads(response.text)
    if not isinstance(dados, list):
        return []

    validos = []
    for item in dados:
        try:
            ajustado = _ajustar_para_limites(
                item,
                duration_video,
                min_duracao,
                max_duracao,
            )
            if ajustado:
                validos.append(ajustado)
        except Exception:
            continue

    validos.sort(key=lambda x: int(x.get("virality_score", 0)), reverse=True)

    escolhidos = []
    for item in validos:
        sobrepoe = False
        for existente in escolhidos:
            if item["start_seconds"] < existente["end_seconds"] and item["end_seconds"] > existente["start_seconds"]:
                sobrepoe = True
                break
        if not sobrepoe:
            escolhidos.append(item)
        if len(escolhidos) >= max_cortes:
            break

    return escolhidos
