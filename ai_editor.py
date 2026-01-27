import google.generativeai as genai
import json
import typing_extensions as typing
import math

# --- CONFIGURAÇÃO ---
API_KEY = "AIzaSyBlf52M5Gm6IQRB_Xcud_YNScsAg2b5vhQ"

genai.configure(api_key=API_KEY)

class ViralSegment(typing.TypedDict):
    start_time: str
    end_time: str
    title: str
    virality_score: int
    reason: str

def formatar_tempo(segundos):
    m = math.floor(segundos / 60)
    s = math.floor(segundos % 60)
    return f"{m:02d}:{s:02d}"

def analisar_video_com_gemini(segments, duration_video):
    print("   🧠 Preparando transcrição com timestamps para o Gemini...")
    
    # Monta um texto onde cada frase tem seu tempo exato
    # Ex: [02:30] O policial falou isso.
    transcript_with_time = ""
    for seg in segments:
        start_fmt = formatar_tempo(seg['start'])
        text = seg['text'].strip()
        transcript_with_time += f"[{start_fmt}] {text}\n"

    print("   🚀 Enviando para análise inteligente...")

    modelos_para_tentar = [
        'gemini-2.0-flash',
        'gemini-2.5-flash',
        'gemini-1.5-pro-latest'
    ]
    
    model = None
    for nome_modelo in modelos_para_tentar:
        try:
            modelo_teste = genai.GenerativeModel(nome_modelo)
            modelo_teste.generate_content("Oi")
            model = modelo_teste
            print(f"      ✅ Modelo conectado: {nome_modelo}")
            break
        except:
            continue
            
    if not model:
        print("   ❌ ERRO: Nenhum modelo disponível.")
        return []

    # PROMPT ATUALIZADO: Pede 2 a 3 minutos e exige precisão
    prompt = f"""
    Aja como um editor de vídeo expert em cortes virais (Podcasts, Entrevistas).
    
    Abaixo está a transcrição de um vídeo com timestamps no formato [MM:SS].
    
    Sua Missão: Encontrar os 3 melhores cortes virais.
    
    REGRAS OBRIGATÓRIAS:
    1. **DURAÇÃO:** Cada corte deve ter entre **2 minutos (120s) e 3 minutos (180s)**.
    2. **PRECISÃO:** O corte deve começar EXATAMENTE no início de uma frase e terminar EXATAMENTE no fim de um raciocínio. Não corte palavras pela metade.
    3. **CONTEXTO:** O corte deve contar uma história completa (começo, meio e fim) dentro desse tempo.
    4. **Output:** Retorne os timestamps no formato "MM:SS" (ex: "02:15").
    
    Transcrição (trecho):
    "{transcript_with_time[:60000]}"
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=list[ViralSegment]
            )
        )
        
        segments = json.loads(response.text)
        print(f"   ✨ Gemini encontrou {len(segments)} cortes longos!")
        return segments

    except Exception as e:
        print(f"   ❌ Erro na IA: {e}")
        return []