import json
import os
import re
import random
import time

from google import genai
from google.genai import types

def formatar_tempo(segundos: float) -> str:
    segundos = max(0, int(segundos))
    h = segundos // 3600
    m = (segundos % 3600) // 60
    s = segundos % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def tempo_para_segundos(valor) -> float:
    if isinstance(valor, (int, float)):
        return float(valor)
    partes = [float(p) for p in str(valor).strip().split(":")]
    if len(partes) == 2:
        return partes[0] * 60 + partes[1]
    if len(partes) == 3:
        return partes[0] * 3600 + partes[1] * 60 + partes[2]
    raise ValueError(f"Timestamp inválido: {valor}")

def _configurar_cliente():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Defina a variável de ambiente GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)

def _modelo():
    return os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

def _is_transient_gemini_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in (
        "429", "500", "502", "503", "504",
        "resource_exhausted", "unavailable", "deadline_exceeded",
        "high demand", "temporarily", "timeout",
    ))


def _gerar_json(client, prompt: str):
    max_attempts = max(1, int(os.getenv("GEMINI_MAX_ATTEMPTS", "6")))
    base_delay = max(1.0, float(os.getenv("GEMINI_RETRY_BASE_SECONDS", "3")))

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=_modelo(),
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini não retornou conteúdo.")
            return _parse_json_response(response.text)
        except Exception as exc:
            if not _is_transient_gemini_error(exc) or attempt >= max_attempts:
                if _is_transient_gemini_error(exc):
                    raise RuntimeError(
                        "O Gemini está temporariamente sobrecarregado. "
                        f"Foram feitas {max_attempts} tentativas automáticas. "
                        "Tente novamente em alguns minutos."
                    ) from exc
                raise

            delay = min(45.0, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, min(2.0, delay * 0.2))
            print(
                f"Gemini temporariamente indisponível "
                f"(tentativa {attempt}/{max_attempts}). "
                f"Nova tentativa em {delay:.1f}s..."
            )
            time.sleep(delay)

def _parse_json_response(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise

def _segment_text(seg):
    return f"[{formatar_tempo(seg['start'])} - {formatar_tempo(seg['end'])}] {seg['text'].strip()}"

def _build_chunks(segments, chunk_seconds=900, overlap_seconds=90):
    if not segments:
        return []
    chunks = []
    start = 0.0
    max_end = max(float(s["end"]) for s in segments)
    while start < max_end:
        end = start + chunk_seconds
        selected = [s for s in segments if float(s["end"]) >= start and float(s["start"]) <= end]
        if selected:
            chunks.append(selected)
        start += max(60, chunk_seconds - overlap_seconds)
    return chunks

def _closest_word_boundary(segments, target: float, mode: str) -> float:
    words = [word for seg in segments for word in (seg.get("words") or [])]
    if not words:
        return target
    if mode == "start":
        candidates = [w["start"] for w in words if target - 2.0 <= w["start"] <= target + 0.5]
        return min(candidates, key=lambda x: abs(x - target)) if candidates else target
    candidates = [w["end"] for w in words if target - 0.5 <= w["end"] <= target + 2.0]
    return min(candidates, key=lambda x: abs(x - target)) if candidates else target

def _normalize_candidate(item, segments, duration_video, min_duracao, max_duracao):
    inicio = max(0.0, tempo_para_segundos(item.get("start_time")))
    fim = min(float(duration_video), tempo_para_segundos(item.get("end_time")))
    inicio = _closest_word_boundary(segments, inicio, "start")
    fim = _closest_word_boundary(segments, fim, "end")
    if fim <= inicio:
        return None
    duracao = fim - inicio
    if duracao < min_duracao or duracao > max_duracao:
        return None
    return {
        "start_time": formatar_tempo(inicio),
        "end_time": formatar_tempo(fim),
        "start_seconds": round(inicio, 3),
        "end_seconds": round(fim, 3),
        "duration": round(duracao, 3),
        "title": str(item.get("title") or "Corte").strip()[:120],
        "virality_score": max(0, min(100, int(item.get("virality_score", 0)))),
        "context_score": max(0, min(100, int(item.get("context_score", 0)))),
        "reason": str(item.get("reason") or "").strip()[:500],
    }

def _ask_candidates(client, chunk, min_duracao, max_duracao, limit=5):
    transcript = "\n".join(_segment_text(s) for s in chunk)
    prompt = f"""
Você é um editor profissional de podcasts, entrevistas e vídeos longos.
Analise SOMENTE a transcrição abaixo e proponha até {limit} candidatos de corte.
Cada candidato deve funcionar sozinho, com contexto suficiente para quem não viu o vídeo original.

Regras:
- duração entre {min_duracao} e {max_duracao} segundos;
- começar no início de uma ideia e terminar após sua conclusão natural;
- evitar introduções vazias e trechos sem payoff;
- priorizar história, revelação, conflito, humor, emoção, opinião forte, dica útil ou explicação surpreendente;
- não inventar conteúdo e não cortar no meio de frase;
- usar timestamps existentes;
- dar virality_score e context_score de 0 a 100;
- retornar APENAS JSON válido.

Formato:
[
  {{
    "start_time": "00:10:20",
    "end_time": "00:11:40",
    "title": "Título curto",
    "virality_score": 85,
    "context_score": 92,
    "reason": "Por que funciona como corte isolado."
  }}
]

TRANSCRIÇÃO:
{transcript}
"""
    data = _gerar_json(client, prompt)
    return data if isinstance(data, list) else []

def _rank_final(client, candidates, max_cortes):
    compact = json.dumps(candidates, ensure_ascii=False)
    prompt = f"""
Você é o editor-chefe. Escolha até {max_cortes} melhores cortes entre os candidatos abaixo.

Critérios, em ordem:
1. contexto completo;
2. começo forte sem depender do trecho anterior;
3. conclusão satisfatória;
4. retenção e potencial de compartilhamento;
5. diversidade de assunto;
6. evitar cortes parecidos ou sobrepostos.

Retorne APENAS um array JSON com os índices escolhidos, na melhor ordem.
Exemplo: [2, 0, 5]

CANDIDATOS:
{compact}
"""
    data = _gerar_json(client, prompt)
    if not isinstance(data, list):
        return []
    result = []
    for value in data:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result

def analisar_video_com_gemini(segments, duration_video, max_cortes=3, min_duracao=45, max_duracao=180):
    if not segments:
        return []

    client = _configurar_cliente()
    chunk_seconds = int(os.getenv("AI_CHUNK_SECONDS", "900"))
    overlap_seconds = int(os.getenv("AI_CHUNK_OVERLAP_SECONDS", "90"))

    all_candidates = []
    for chunk in _build_chunks(segments, chunk_seconds, overlap_seconds):
        raw = _ask_candidates(client, chunk, min_duracao, max_duracao, max(5, max_cortes * 2))
        for item in raw:
            try:
                normalized = _normalize_candidate(item, segments, duration_video, min_duracao, max_duracao)
                if normalized:
                    all_candidates.append(normalized)
            except (KeyError, TypeError, ValueError):
                continue

    if not all_candidates:
        return []

    deduped = []
    for cand in sorted(all_candidates, key=lambda x: (x["context_score"], x["virality_score"]), reverse=True):
        duplicate = False
        for existing in deduped:
            overlap = max(0.0, min(cand["end_seconds"], existing["end_seconds"]) - max(cand["start_seconds"], existing["start_seconds"]))
            shorter = min(cand["duration"], existing["duration"])
            if shorter and overlap / shorter > 0.65:
                duplicate = True
                break
        if not duplicate:
            deduped.append(cand)

    shortlist = deduped[:20]
    try:
        order = _rank_final(client, shortlist, max_cortes)
    except Exception:
        order = []

    chosen = []
    for idx in order:
        if 0 <= idx < len(shortlist) and shortlist[idx] not in chosen:
            chosen.append(shortlist[idx])
        if len(chosen) >= max_cortes:
            break

    if len(chosen) < max_cortes:
        for candidate in shortlist:
            if candidate not in chosen:
                chosen.append(candidate)
            if len(chosen) >= max_cortes:
                break
    return chosen
