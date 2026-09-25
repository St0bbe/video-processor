import json
import os
import re
from typing import List

import google.generativeai as genai

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

def _configurar_modelo():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Defina a variável de ambiente GEMINI_API_KEY.")

    genai.configure(api_key=api_key)
    nome = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return genai.GenerativeModel(nome)

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
        selected = [
            s for s in segments
            if float(s["end"]) >= start and float(s["start"]) <= end
        ]
        if selected:
            chunks.append(selected)
        start += max(60, chunk_seconds - overlap_seconds)

    return chunks

def _closest_word_boundary(segments, target: float, mode: str) -> float:
    words = []
    for seg in segments:
        for word in seg.get("words") or []:
            words.append(word)

    if not words:
        return target

    if mode == "start":
        candidates = [w["start"] for w in words if w["start"] <= target + 2.0]
        return max(candidates) if candidates else target

    candidates = [w["end"] for w in words if w["end"] >= target - 2.0]
    return min(candidates) if candidates else target

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

def _ask_candidates(model, chunk, min_duracao, max_duracao, limit=5):
    transcript = "\n".join(_segment_text(s) for s in chunk)
    prompt = f"""
Você é um editor profissional de podcasts, entrevistas e vídeos longos.

Analise SOMENTE a transcrição abaixo e proponha até {limit} candidatos de corte.
Cada candidato deve funcionar sozinho, com contexto suficiente para quem não viu o vídeo original.

Regras:
- duração entre {min_duracao} e {max_duracao} segundos;
- começar no início de uma ideia;
- terminar depois da conclusão natural;
- evitar introduções vazias, cumprimentos e partes sem payoff;
- priorizar história completa, revelação, conflito, humor, emoção, opinião forte, dica útil ou explicação surpreendente;
- não inventar nada;
- não cortar no meio de uma frase;
- usar timestamps existentes;
- dar virality_score de 0 a 100;
- dar context_score de 0 a 100;
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
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(response_mime_type="application/json"),
    )
    data = _parse_json_response(response.text)
    return data if isinstance(data, list) else []

def _rank_final(model, candidates, max_cortes):
    compact = json.dumps(candidates, ensure_ascii=False)
    prompt = f"""
Você é o editor-chefe. Abaixo estão candidatos de cortes já validados.

Escolha até {max_cortes} melhores cortes no conjunto inteiro.
Critérios, em ordem:
1. contexto completo;
2. começo forte sem depender do trecho anterior;
3. conclusão satisfatória;
4. potencial de retenção/compartilhamento;
5. diversidade de assunto;
6. evitar cortes muito parecidos ou sobrepostos.

Retorne APENAS um array JSON com os índices dos escolhidos, na melhor ordem.
Exemplo: [2, 0, 5]

CANDIDATOS:
{compact}
"""
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(response_mime_type="application/json"),
    )
    data = _parse_json_response(response.text)
    if not isinstance(data, list):
        return []
    return [int(i) for i in data if isinstance(i, (int, float, str)) and str(i).isdigit()]

def analisar_video_com_gemini(
    segments,
    duration_video,
    max_cortes=3,
    min_duracao=45,
    max_duracao=180,
):
    if not segments:
        return []

    model = _configurar_modelo()
    chunk_seconds = int(os.getenv("AI_CHUNK_SECONDS", "900"))
    overlap_seconds = int(os.getenv("AI_CHUNK_OVERLAP_SECONDS", "90"))

    all_candidates = []
    for chunk in _build_chunks(segments, chunk_seconds, overlap_seconds):
        raw = _ask_candidates(
            model,
            chunk,
            min_duracao=min_duracao,
            max_duracao=max_duracao,
            limit=max(5, max_cortes * 2),
        )
        for item in raw:
            try:
                normalized = _normalize_candidate(
                    item,
                    segments,
                    duration_video,
                    min_duracao,
                    max_duracao,
                )
                if normalized:
                    all_candidates.append(normalized)
            except Exception:
                continue

    if not all_candidates:
        return []

    deduped = []
    for cand in sorted(
        all_candidates,
        key=lambda x: (x["context_score"], x["virality_score"]),
        reverse=True,
    ):
        duplicate = False
        for existing in deduped:
            overlap = max(
                0.0,
                min(cand["end_seconds"], existing["end_seconds"])
                - max(cand["start_seconds"], existing["start_seconds"]),
            )
            shorter = min(cand["duration"], existing["duration"])
            if shorter and overlap / shorter > 0.65:
                duplicate = True
                break
        if not duplicate:
            deduped.append(cand)

    shortlist = deduped[: min(20, len(deduped))]
    order = _rank_final(model, shortlist, max_cortes)

    chosen = []
    for idx in order:
        if 0 <= idx < len(shortlist):
            candidate = shortlist[idx]
            if candidate not in chosen:
                chosen.append(candidate)
        if len(chosen) >= max_cortes:
            break

    if len(chosen) < max_cortes:
        for candidate in shortlist:
            if candidate not in chosen:
                chosen.append(candidate)
            if len(chosen) >= max_cortes:
                break

    return chosen
